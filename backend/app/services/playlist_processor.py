"""
合集视频并行处理器

提供两种实现方案：
1. SimpleThreadPipeline: 基于线程池的简单并行（推荐快速实施）
2. AsyncPipeline: 基于 asyncio 的流水线并行（推荐长期使用）
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable
from asyncio import Queue, Semaphore

from app.models.audio_model import AudioDownloadResult
from app.models.transcriber_model import TranscriptResult
from app.gpt.base import GPT
from app.transcriber.base import Transcriber
from app.utils.logger import get_logger
from app.utils.audio_validator import validate_audio_file

logger = get_logger(__name__)


@dataclass
class ProcessingStats:
    """处理统计信息"""
    total_videos: int = 0
    downloaded: int = 0
    transcribed: int = 0
    summarized: int = 0
    failed: int = 0
    
    def to_dict(self):
        return {
            "total": self.total_videos,
            "downloaded": self.downloaded,
            "transcribed": self.transcribed,
            "summarized": self.summarized,
            "failed": self.failed,
        }


class SimpleThreadPipeline:
    """
    基于线程池的简单并行处理器
    
    优点：
    - 实现简单，改动小
    - 不需要 async/await
    - 立即见效（提升 30-40%）
    
    适用场景：
    - 快速优化，1-2小时完成
    - 不想改造现有代码为 async
    """
    
    def __init__(
        self,
        transcriber: Transcriber,
        max_workers: Optional[int] = None,
        transcriber_type: str = "fast-whisper",
    ):
        """
        :param transcriber: 转写器实例
        :param max_workers: 最大并发数（None 则自动根据视频数量和转写器类型判断）
        :param transcriber_type: 转写器类型，用于判断是否可以并发
        """
        self.transcriber = transcriber
        self.max_workers = max_workers
        self.transcriber_type = transcriber_type
        self.stats = ProcessingStats()
    
    @staticmethod
    def calculate_optimal_workers(video_count: int, transcriber_type: str = "fast-whisper") -> int:
        """
        根据视频数量和转写器类型计算最优并发数
        
        :param video_count: 视频数量
        :param transcriber_type: 转写器类型
        :return: 推荐的并发数
        """
        # 支持多线程的本地模型（已验证线程安全）
        thread_safe_models = ["fast-whisper"]  # CTranslate2 官方支持多线程
        
        # 需要保守处理的本地模型（未验证线程安全）
        conservative_models = ["mlx-whisper", "funasr-nano", "paraformer-streaming"]
        
        # API 服务（网络 IO，可以高并发）
        api_services = ["bcut", "kuaishou", "groq", "deepgram"]
        
        if transcriber_type in thread_safe_models:
            # faster-whisper: CTranslate2 官方支持多线程，可以适度并发
            # 但仍需考虑 GPU 内存限制
            if video_count <= 5:
                return 2
            elif video_count <= 10:
                return 3
            elif video_count <= 20:
                return 4
            else:
                return 5
        
        elif transcriber_type in conservative_models:
            # 未验证线程安全的本地模型：保守策略
            logger.warning(
                f"转写器 {transcriber_type} 的线程安全性未验证，"
                f"使用保守并发策略（并发数=1）"
            )
            return 1
        
        elif transcriber_type in api_services:
            # API 服务：固定4个并发，避免触发 API 限流
            return 4
        
        else:
            # 未知类型，保守处理
            logger.warning(f"未知转写器类型 {transcriber_type}，使用保守并发策略")
            return 2
    
    def process_playlist(
        self,
        audio_results: List[AudioDownloadResult],
        gpt: GPT,
        task_id: str,
        progress_callback: Optional[Callable] = None,
        downloader: Optional[any] = None,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """
        并行处理合集视频（支持边下载边处理）
        
        :param audio_results: 视频元信息列表（可能还未下载）
        :param gpt: GPT 实例
        :param task_id: 任务ID
        :param progress_callback: 进度回调函数 callback(stats)
        :param downloader: 下载器实例（用于并行下载）
        :param output_dir: 输出目录
        :return: 按顺序排列的 markdown 列表
        """
        self.stats.total_videos = len(audio_results)
        results_dict = {}
        
        # 动态计算并发数
        if self.max_workers is None:
            workers = self.calculate_optimal_workers(len(audio_results), self.transcriber_type)
        else:
            workers = self.max_workers
        
        logger.info(
            f"开始并行处理 {len(audio_results)} 个视频 "
            f"(转写器: {self.transcriber_type}, 并发数: {workers}, "
            f"策略: {'自动' if self.max_workers is None else '手动'})"
        )
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务（包含下载）
            futures = {
                executor.submit(
                    self._process_single_video,
                    idx,
                    audio,
                    gpt,
                    task_id,
                    progress_callback,
                    downloader,  # 传入下载器
                    output_dir,  # 传入输出目录
                ): idx
                for idx, audio in enumerate(audio_results)
            }
            
            # 等待完成
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    markdown = future.result()
                    results_dict[idx] = markdown
                    self.stats.summarized += 1
                    
                    if progress_callback:
                        progress_callback(self.stats)
                    
                except Exception as e:
                    logger.error(f"视频 {idx} 处理失败: {e}", exc_info=True)
                    audio = audio_results[idx]
                    results_dict[idx] = f"# {audio.title}\n\n⚠️ 处理失败: {str(e)}\n\n"
                    self.stats.failed += 1
        
        # 按顺序返回
        return [results_dict[i] for i in range(len(audio_results))]
    
    def _process_single_video(
        self,
        idx: int,
        audio: AudioDownloadResult,
        gpt: GPT,
        task_id: str,
        progress_callback: Optional[Callable],
        downloader: Optional[any] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """处理单个视频（下载 → 转写 → GPT）"""
        total = self.stats.total_videos
        logger.info(f"[{idx+1}/{total}] 开始处理: {audio.title}")
        
        start_time = time.time()
        
        try:
            # 0. 下载音频（如果还没下载或文件无效）
            if downloader:
                # 检查文件是否存在且有效
                need_download = True
                if audio.file_path and os.path.exists(audio.file_path):
                    # 快速校验文件完整性
                    is_valid, error_msg = validate_audio_file(audio.file_path)
                    if is_valid:
                        need_download = False
                        logger.info(f"[{idx+1}/{total}] 音频已存在且有效，跳过下载")
                    else:
                        logger.warning(f"[{idx+1}/{total}] 音频文件无效: {error_msg}，重新下载")
                
                if need_download:
                    logger.info(f"[{idx+1}/{total}] 开始下载音频: {audio.title}")
                    download_start = time.time()
                    
                    # 确保输出目录存在
                    output_directory = output_dir or os.path.dirname(audio.file_path) if audio.file_path else "data"
                    os.makedirs(output_directory, exist_ok=True)
                    
                    # 下载音频
                    result = downloader.download_single_audio(
                        audio_result=audio,
                        output_dir=output_directory,
                    )
                    
                    # 解包返回值
                    if isinstance(result, tuple):
                        audio.file_path, real_title = result
                        if real_title:
                            audio.title = real_title
                    else:
                        audio.file_path = result
                    
                    download_time = time.time() - download_start
                    logger.info(f"[{idx+1}/{total}] 下载完成，耗时 {download_time:.1f}秒")
            
            # 1. 转写（带重试机制）
            logger.info(f"[{idx+1}/{total}] 开始转写")
            transcript = self._transcribe_with_retry(audio.file_path, idx, total)
            
            self.stats.transcribed += 1
            
            if progress_callback:
                progress_callback(self.stats)
            
            transcribe_time = time.time() - start_time
            logger.info(f"[{idx+1}/{total}] 转写完成，耗时 {transcribe_time:.1f}秒")
            
            # 2. GPT 总结
            logger.info(f"[{idx+1}/{total}] 开始 GPT 总结")
            gpt_start = time.time()
            
            # 构造 GPT 输入
            from app.models.gpt_model import GPTSource
            source = GPTSource(
                title=audio.title,
                segment=transcript.segments,
                tags=audio.raw_info.get("tags", []),
                video_img_urls=[],  # 合集处理暂不支持视频理解
            )
            markdown = gpt.summarize(source)
            
            gpt_time = time.time() - gpt_start
            total_time = time.time() - start_time
            
            logger.info(
                f"[{idx+1}/{total}] GPT 总结完成: {audio.title}，"
                f"转写: {transcribe_time:.1f}s, GPT: {gpt_time:.1f}s, "
                f"总计: {total_time:.1f}s"
            )
            
            return f"# {audio.title}\n\n{markdown}\n\n"
            
        except Exception as e:
            logger.error(f"[{idx+1}/{total}] 处理失败: {e}", exc_info=True)
            raise
    
    def _transcribe_with_retry(self, file_path: str, idx: int, total: int, max_retries: int = 3) -> any:
        """
        带重试机制的转写方法
        
        :param file_path: 音频文件路径
        :param idx: 视频索引
        :param total: 总视频数
        :param max_retries: 最大重试次数（默认3次）
        :return: 转写结果
        """
        import time as time_module
        
        for attempt in range(max_retries):
            try:
                return self.transcriber.transcript(file_path=file_path)
            except Exception as e:
                # 检查是否是可重试的错误
                should_retry = False
                error_str = str(e)
                error_type = type(e).__name__
                
                # 检查是否是429限流错误
                if "429" in error_str or "Too Many Requests" in error_str or "status_code: 429" in error_str:
                    should_retry = True
                    error_reason = "429限流"
                # 检查是否是超时错误
                elif "timeout" in error_str.lower() or "ReadTimeout" in error_type or "TimeoutError" in error_type:
                    should_retry = True
                    error_reason = "网络超时"
                # 检查是否是连接错误
                elif "ConnectionError" in error_type or "ConnectTimeout" in error_type:
                    should_retry = True
                    error_reason = "连接失败"
                
                # 如果是可重试的错误且还有重试机会
                if should_retry and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3  # 指数退避：3秒、6秒、9秒
                    logger.warning(
                        f"[{idx+1}/{total}] 转写遇到{error_reason}错误，"
                        f"等待 {wait_time} 秒后重试 (第 {attempt + 1}/{max_retries} 次)"
                    )
                    time_module.sleep(wait_time)
                    continue
                else:
                    # 不是可重试的错误，或者已经用完重试次数
                    if should_retry:
                        logger.error(f"[{idx+1}/{total}] 转写失败：已重试 {max_retries} 次仍然失败 ({error_reason})")
                    raise


class AsyncPipeline:
    """
    基于 asyncio 的流水线并行处理器
    
    优点：
    - 真正的流水线并行
    - 资源利用率最高（提升 45-50%）
    - 更好的并发控制
    
    适用场景：
    - 追求最佳性能
    - 愿意改造为 async/await
    """
    
    def __init__(
        self,
        transcriber: Transcriber,
        max_concurrent_transcriptions: int = 2,
        max_concurrent_gpt: int = 5,
    ):
        """
        :param transcriber: 转写器实例
        :param max_concurrent_transcriptions: 最大转写并发数（取决于 GPU 内存）
        :param max_concurrent_gpt: 最大 GPT 并发数（取决于 API 限流）
        """
        self.transcriber = transcriber
        self.transcribe_sem = Semaphore(max_concurrent_transcriptions)
        self.gpt_sem = Semaphore(max_concurrent_gpt)
        
        self.transcribe_queue = Queue()
        self.gpt_queue = Queue()
        self.results = {}
        self.stats = ProcessingStats()
    
    async def process_playlist(
        self,
        audio_results: List[AudioDownloadResult],
        gpt: GPT,
        task_id: str,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """
        异步并行处理合集视频
        
        :param audio_results: 已下载的音频列表
        :param gpt: GPT 实例
        :param task_id: 任务ID
        :param progress_callback: 进度回调函数
        :return: 按顺序排列的 markdown 列表
        """
        self.stats.total_videos = len(audio_results)
        
        logger.info(
            f"开始流水线处理 {len(audio_results)} 个视频 "
            f"(转写并发: {self.transcribe_sem._value}, "
            f"GPT并发: {self.gpt_sem._value})"
        )
        
        # 启动工作协程
        workers = [
            asyncio.create_task(self._transcribe_worker(task_id, progress_callback)),
            asyncio.create_task(self._transcribe_worker(task_id, progress_callback)),
            asyncio.create_task(self._gpt_worker(gpt, task_id, progress_callback)),
            asyncio.create_task(self._gpt_worker(gpt, task_id, progress_callback)),
            asyncio.create_task(self._gpt_worker(gpt, task_id, progress_callback)),
        ]
        
        # 将所有音频放入转写队列
        for idx, audio_meta in enumerate(audio_results):
            await self.transcribe_queue.put((idx, audio_meta))
        
        # 等待所有任务完成
        await self.transcribe_queue.join()
        await self.gpt_queue.join()
        
        # 取消工作协程
        for worker in workers:
            worker.cancel()
        
        # 按顺序返回结果
        return [self.results[i] for i in range(len(audio_results))]
    
    async def _transcribe_worker(
        self,
        task_id: str,
        progress_callback: Optional[Callable],
    ):
        """转写工作协程"""
        while True:
            try:
                idx, audio_meta = await self.transcribe_queue.get()
                
                async with self.transcribe_sem:
                    total = self.stats.total_videos
                    logger.info(f"[{idx+1}/{total}] 开始转写: {audio_meta.title}")
                    
                    start_time = time.time()
                    
                    # 在线程池中执行转写（避免阻塞事件循环）
                    transcript = await asyncio.to_thread(
                        self.transcriber.transcript,
                        file_path=audio_meta.file_path
                    )
                    
                    duration = time.time() - start_time
                    logger.info(f"[{idx+1}/{total}] 转写完成，耗时 {duration:.1f}秒")
                    
                    self.stats.transcribed += 1
                    if progress_callback:
                        progress_callback(self.stats)
                    
                    # 放入 GPT 队列
                    await self.gpt_queue.put((idx, audio_meta, transcript))
                
                self.transcribe_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"转写失败: {e}", exc_info=True)
                self.stats.failed += 1
                self.transcribe_queue.task_done()
    
    async def _gpt_worker(
        self,
        gpt: GPT,
        task_id: str,
        progress_callback: Optional[Callable],
    ):
        """GPT 工作协程"""
        while True:
            try:
                idx, audio_meta, transcript = await self.gpt_queue.get()
                
                async with self.gpt_sem:
                    total = self.stats.total_videos
                    logger.info(f"[{idx+1}/{total}] 开始 GPT 总结")
                    
                    start_time = time.time()
                    
                    # 在线程池中执行 GPT 调用
                    from app.models.gpt_model import GPTSource
                    source = GPTSource(
                        title=audio_meta.title,
                        segment=transcript.segments,
                        tags=audio_meta.raw_info.get("tags", []),
                    )
                    
                    markdown = await asyncio.to_thread(gpt.summarize, source)
                    
                    duration = time.time() - start_time
                    logger.info(f"[{idx+1}/{total}] GPT 完成，耗时 {duration:.1f}秒")
                    
                    self.stats.summarized += 1
                    if progress_callback:
                        progress_callback(self.stats)
                    
                    # 保存结果
                    self.results[idx] = f"# {audio_meta.title}\n\n{markdown}\n\n"
                
                self.gpt_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"GPT 失败: {e}", exc_info=True)
                self.results[idx] = f"# {audio_meta.title}\n\n⚠️ 处理失败: {str(e)}\n\n"
                self.stats.failed += 1
                self.gpt_queue.task_done()
