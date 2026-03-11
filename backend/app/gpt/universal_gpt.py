from app.gpt.base import GPT
from app.gpt.prompt_builder import generate_base_prompt
from app.models.gpt_model import GPTSource
from app.gpt.prompt import BASE_PROMPT, AI_SUM, SCREENSHOT, LINK
from app.gpt.utils import fix_markdown
from app.models.transcriber_model import TranscriptSegment
from datetime import timedelta
from typing import List
import os


class UniversalGPT(GPT):
    def __init__(self, client, model: str, temperature: float = 0.7):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.screenshot = False
        self.link = False

    def _format_time(self, seconds: float) -> str:
        return str(timedelta(seconds=int(seconds)))[2:]

    def _build_segment_text(self, segments: List[TranscriptSegment]) -> str:
        segment_texts = []
        for seg in segments:
            text = seg.text.strip()
            time_str = self._format_time(seg.start)
            segment_text = f"{time_str} - {text}"
            segment_texts.append(segment_text)
        
        return "\n".join(segment_texts)

    def ensure_segments_type(self, segments) -> List[TranscriptSegment]:
        return [TranscriptSegment(**seg) if isinstance(seg, dict) else seg for seg in segments]

    def create_messages(self, segments: List[TranscriptSegment], **kwargs):

        content_text = generate_base_prompt(
            title=kwargs.get('title'),
            segment_text=self._build_segment_text(segments),
            tags=kwargs.get('tags'),
            _format=kwargs.get('_format'),
            style=kwargs.get('style'),
            extras=kwargs.get('extras'),
        )

        video_img_urls = kwargs.get('video_img_urls', [])
        
        # 如果没有图片，使用简单的文本格式
        if not video_img_urls:
            messages = [{
                "role": "user",
                "content": content_text
            }]
            return messages

        # ⛳ 组装 content 数组，支持 text + image_url 混合
        content = [{"type": "text", "text": content_text}]

        for url in video_img_urls:
            # 确保 URL 是有效的
            if url and isinstance(url, str):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                        "detail": "auto"
                    }
                })

        #  正确格式：整体包在一个 message 里，role + content array
        messages = [{
            "role": "user",
            "content": content
        }]

        return messages

    def list_models(self):
        return self.client.models.list()

    def summarize(self, source: GPTSource) -> str:
        self.screenshot = source.screenshot
        self.link = source.link
        source.segment = self.ensure_segments_type(source.segment)

        # 检查是否包含图片
        has_images = source.video_img_urls and len(source.video_img_urls) > 0
        
        if has_images:
            from app.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info(f"📸 包含 {len(source.video_img_urls)} 张视频截图")
            
            # 如果图片数量超过阈值，使用分批处理
            max_images_per_batch = int(os.getenv("MAX_IMAGES_PER_BATCH", "8"))
            if len(source.video_img_urls) > max_images_per_batch:
                logger.info(f"🔄 图片数量超过 {max_images_per_batch} 张，启用分批处理")
                return self._summarize_with_batches(source, max_images_per_batch)
        
        # 单次请求处理
        messages = self.create_messages(
            source.segment,
            title=source.title,
            tags=source.tags,
            video_img_urls=source.video_img_urls,
            _format=source._format,
            style=source.style,
            extras=source.extras
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            from app.utils.logger import get_logger
            logger = get_logger(__name__)
            error_msg = str(e)
            
            # 如果包含图片时出错，尝试降级处理
            if has_images and ("image" in error_msg.lower() or "400" in error_msg or "request" in error_msg.lower()):
                logger.warning(f"⚠️ Vision API 调用失败: {error_msg}")
                logger.warning(f"   尝试分批处理...")
                
                try:
                    # 尝试分批处理
                    return self._summarize_with_batches(source, batch_size=5)
                except Exception as batch_error:
                    logger.warning(f"⚠️ 分批处理也失败，降级为纯文本模式: {batch_error}")
                    
                    # 最后降级为纯文本模式
                    messages_text_only = self.create_messages(
                        source.segment,
                        title=source.title,
                        tags=source.tags,
                        video_img_urls=[],
                        _format=source._format,
                        style=source.style,
                        extras=source.extras
                    )
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages_text_only,
                        temperature=0.7
                    )
                    logger.info("✅ 纯文本模式调用成功")
                    return response.choices[0].message.content.strip()
            else:
                raise
    
    def _summarize_with_batches(self, source: GPTSource, batch_size: int = 8) -> str:
        """
        分批处理视频截图，每批包含部分图片和对应时间段的转写文本
        最后将所有批次的总结合并成完整笔记
        
        :param source: GPT 数据源
        :param batch_size: 每批最多包含的图片数量
        :return: 合并后的完整 Markdown
        """
        from app.utils.logger import get_logger
        logger = get_logger(__name__)
        
        video_img_urls = source.video_img_urls or []
        segments = source.segment
        total_batches = (len(video_img_urls) + batch_size - 1) // batch_size
        
        logger.info(f"📦 开始分批处理：共 {len(video_img_urls)} 张图片，分为 {total_batches} 批")
        
        batch_summaries = []
        
        # 计算每批对应的时间范围
        total_duration = segments[-1].end if segments else 0
        time_per_batch = total_duration / total_batches if total_batches > 0 else 0
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(video_img_urls))
            batch_images = video_img_urls[start_idx:end_idx]
            
            # 计算这批图片对应的时间范围
            start_time = batch_idx * time_per_batch
            end_time = (batch_idx + 1) * time_per_batch
            
            # 筛选对应时间段的转写文本
            batch_segments = [
                seg for seg in segments 
                if start_time <= seg.start < end_time
            ] if segments else []
            
            # 如果没有对应的转写片段，使用全部转写（避免丢失信息）
            if not batch_segments:
                batch_segments = segments
            
            logger.info(f"  批次 {batch_idx + 1}/{total_batches}: {len(batch_images)} 张图片, {len(batch_segments)} 个转写片段")
            
            # 创建这一批的消息
            messages = self.create_messages(
                batch_segments,
                title=f"{source.title} - 第 {batch_idx + 1} 部分",
                tags=source.tags,
                video_img_urls=batch_images,
                _format=source._format,
                style=source.style,
                extras=f"这是视频的第 {batch_idx + 1}/{total_batches} 部分。{source.extras or ''}"
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7
                )
                batch_summary = response.choices[0].message.content.strip()
                batch_summaries.append(batch_summary)
                logger.info(f"  ✅ 批次 {batch_idx + 1} 处理完成")
            except Exception as e:
                logger.error(f"  ❌ 批次 {batch_idx + 1} 处理失败: {e}")
                # 失败时使用纯文本模式
                messages_text = self.create_messages(
                    batch_segments,
                    title=f"{source.title} - 第 {batch_idx + 1} 部分",
                    tags=source.tags,
                    video_img_urls=[],
                    _format=source._format,
                    style=source.style,
                    extras=f"这是视频的第 {batch_idx + 1}/{total_batches} 部分。{source.extras or ''}"
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages_text,
                    temperature=0.7
                )
                batch_summaries.append(response.choices[0].message.content.strip())
        
        # 如果只有一批，直接返回
        if len(batch_summaries) == 1:
            return batch_summaries[0]
        
        # 合并所有批次的总结
        logger.info(f"🔗 合并 {len(batch_summaries)} 个批次的总结...")
        merged_content = "\n\n---\n\n".join(batch_summaries)
        
        # 让 AI 对合并后的内容进行最终整理
        final_prompt = f"""以下是对视频《{source.title}》分段总结的内容，请将它们整合成一篇完整、连贯的笔记。

要求：
1. 保持所有重要信息和细节
2. 去除重复内容
3. 确保逻辑连贯、结构清晰
4. 保持原有的 Markdown 格式

分段内容：

{merged_content}
"""
        
        try:
            final_messages = [{"role": "user", "content": final_prompt}]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=final_messages,
                temperature=0.7
            )
            final_summary = response.choices[0].message.content.strip()
            logger.info("✅ 分批处理完成，已合并为完整笔记")
            return final_summary
        except Exception as e:
            logger.warning(f"⚠️ 最终合并失败，返回拼接结果: {e}")
            return merged_content
