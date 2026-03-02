import os
from typing import Optional

from app.decorators.timeit import timeit
from app.models.transcriber_model import TranscriptResult, TranscriptSegment
from app.transcriber.base import Transcriber
from app.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)


class DeepgramTranscriber(Transcriber):
    """
    Deepgram Whisper 语音识别转写器
    
    特点:
        - 使用 Whisper 模型，对中文支持更好
        - 速度快，准确率高
        - 支持说话人分离（Diarization）
        - 支持智能格式化
    
    注意：
        - Nova-3 不支持中文
        - Nova-2 对中文支持较差
        - Whisper 模型对中文识别质量更好
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 DEEPGRAM_API_KEY，请在 .env 文件中配置")
        
        try:
            from deepgram import DeepgramClient
            self.DeepgramClient = DeepgramClient
            logger.info("Deepgram SDK 初始化成功")
        except ImportError:
            raise ImportError(
                "请安装 Deepgram SDK: pip install deepgram-sdk"
            )

    @timeit
    def transcript(self, file_path: str) -> TranscriptResult:
        """
        使用 Deepgram Whisper 进行语音转写
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            TranscriptResult: 转写结果
        """
        import time
        start_time = time.perf_counter()
        
        try:
            logger.info(f"🎤 开始音频转写: {file_path}")
            
            # 对于中文，使用 Whisper 模型（Nova-2 对中文支持不够好）
            model = "whisper-large"  # Whisper 对中文支持更好
            logger.info(f"使用模型: Deepgram {model}")
            
            # 初始化 Deepgram 客户端
            client = self.DeepgramClient(api_key=self.api_key)
            
            # 读取音频文件
            with open(file_path, "rb") as audio_file:
                buffer_data = audio_file.read()
            
            # 调用 Deepgram API - 使用 Whisper 模型
            response = client.listen.v1.media.transcribe_file(
                request=buffer_data,
                model=model,
                language="zh",
                smart_format=True,
                punctuate=True,
                paragraphs=True,
                diarize=False,
            )
            
            # 解析结果 - response 是 Pydantic 对象，直接访问属性
            if not response.results or not response.results.channels:
                raise Exception("Deepgram 返回结果为空")
            
            channel = response.results.channels[0]
            if not channel.alternatives:
                raise Exception("未找到转写结果")
            
            alternative = channel.alternatives[0]
            full_text = alternative.transcript.strip() if alternative.transcript else ""
            
            # 提取分段信息（优先使用 paragraphs，而不是 utterances）
            segments = []
            
            # 尝试使用 paragraphs
            paragraphs_data = None
            if hasattr(alternative, 'paragraphs') and alternative.paragraphs:
                paragraphs_data = alternative.paragraphs
                if hasattr(paragraphs_data, 'paragraphs') and paragraphs_data.paragraphs:
                    logger.info(f"找到 {len(paragraphs_data.paragraphs)} 个段落")
                    for para in paragraphs_data.paragraphs:
                        # 每个段落包含多个句子
                        if hasattr(para, 'sentences') and para.sentences:
                            for sentence in para.sentences:
                                segments.append(TranscriptSegment(
                                    start=sentence.start if hasattr(sentence, 'start') else 0.0,
                                    end=sentence.end if hasattr(sentence, 'end') else 0.0,
                                    text=sentence.text.strip() if hasattr(sentence, 'text') else "",
                                ))
            
            # 如果没有 paragraphs，使用 words 构建分段
            if not segments:
                words = alternative.words if hasattr(alternative, 'words') and alternative.words else []
                if words:
                    logger.info(f"使用 {len(words)} 个词构建分段（每 20 个词一组）")
                    # 每 20 个词一组（中文一个词通常是一个字）
                    chunk_size = 20
                    for i in range(0, len(words), chunk_size):
                        chunk = words[i:i + chunk_size]
                        text = "".join(w.word for w in chunk if hasattr(w, 'word')).strip()
                        if text:
                            segments.append(TranscriptSegment(
                                start=chunk[0].start if hasattr(chunk[0], 'start') else 0.0,
                                end=chunk[-1].end if hasattr(chunk[-1], 'end') else 0.0,
                                text=text,
                            ))
                else:
                    # 没有时间戳信息，使用完整文本
                    logger.warning("未找到时间戳信息，使用完整文本")
                    segments.append(TranscriptSegment(
                        start=0.0,
                        end=0.0,
                        text=full_text,
                    ))
            
            # 检测语言
            detected_language = channel.detected_language if hasattr(channel, 'detected_language') else "zh"
            
            transcript_result = TranscriptResult(
                language=detected_language,
                full_text=full_text,
                segments=segments,
                raw=response.to_dict() if hasattr(response, 'to_dict') else {},
            )
            
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"✅ 音频转写完成，耗时 {elapsed_time:.2f} 秒")
            logger.info(f"识别语言: {detected_language}, 文本长度: {len(full_text)}, 片段数: {len(segments)}")
            
            return transcript_result
            
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(f"❌ Deepgram 转写失败，耗时 {elapsed_time:.2f} 秒: {e}")
            raise

    def on_finish(self, video_path: str, result: TranscriptResult) -> None:
        """转写完成回调"""
        logger.info(f"Deepgram 转写完成: {video_path}")
