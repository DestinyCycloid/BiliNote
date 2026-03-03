import os
import time
import numpy as np
from typing import Optional, List
import threading

from app.decorators.timeit import timeit
from app.models.transcriber_model import TranscriptResult, TranscriptSegment
from app.transcriber.base import Transcriber
from app.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)


class ParaformerStreamingTranscriber(Transcriber):
    """
    FunASR Paraformer-streaming 语音识别转写器
    
    特点:
        - 支持流式转写
        - 220M 参数，轻量级
        - 延迟可配置（300ms-600ms）
        - 专为中文优化
        - 60,000 小时训练数据
    """

    def __init__(
        self, 
        chunk_size: List[int] = None,
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
        use_vad: bool = False,
        use_punc: bool = False,
        device: str = "cuda"
    ):
        """
        初始化 Paraformer-streaming 转写器
        
        Args:
            chunk_size: 流式配置，格式 [0, N, M]
                - N: 每个块的大小（单位：60ms），例如 N=10 表示 600ms
                - 建议值：
                  * [0, 10, 5] = 600ms（实时直播，低延迟）
                  * [0, 17, 8] = 1020ms（录制文件，平衡性能）
                  * [0, 34, 17] = 2040ms（长音频，更快处理）
            encoder_chunk_look_back: 编码器回看块数
            decoder_chunk_look_back: 解码器回看块数
            use_vad: 是否使用 VAD（语音活动检测）- 流式模式下不兼容
            use_punc: 是否使用标点恢复 - 流式模式下不工作
            device: 运行设备
        """
        # 默认使用 2 秒块大小（适合离线文件处理，更快）
        self.chunk_size = chunk_size or [0, 34, 17]  # 2040ms ≈ 2秒
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back
        self.use_vad = use_vad
        self.use_punc = use_punc
        
        # 自动检测设备
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，将使用 CPU 运行")
            logger.info("提示：AMD GPU 需要安装 PyTorch ROCm 版本才能使用 GPU 加速")
            self.device = "cpu"
        else:
            self.device = device
        
        self.model = None
        self._model_lock = threading.Lock()  # 线程锁，保护模型加载
        
        # 检查依赖
        try:
            import funasr
            import soundfile
            logger.info(f"FunASR 版本: {funasr.__version__}")
        except ImportError as e:
            raise ImportError(
                f"请安装必要的依赖: {e}\n"
                "pip install funasr soundfile"
            )

    def _load_model(self):
        """延迟加载模型（线程安全）"""
        # 双重检查锁定模式
        if self.model is None:
            with self._model_lock:
                # 再次检查，避免多个线程重复加载
                if self.model is None:
                    from funasr import AutoModel
                    
                    logger.info("正在加载 Paraformer-streaming 模型...")
            
            # 优先使用本地模型
            local_model_dir = "./models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
            
            if os.path.exists(local_model_dir):
                model_path = local_model_dir
                logger.info(f"使用本地模型: {model_path}")
            else:
                model_path = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
                logger.info(f"本地模型不存在，从 ModelScope 下载: {model_path}")
            
            # 构建模型参数
            model_kwargs = {
                "model": model_path,
                "device": self.device,
                "disable_update": True,
            }
            
            # 添加 VAD
            if self.use_vad:
                vad_model_dir = "./models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
                if os.path.exists(vad_model_dir):
                    model_kwargs["vad_model"] = vad_model_dir
                else:
                    model_kwargs["vad_model"] = "fsmn-vad"
                model_kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
                logger.warning("⚠️ VAD 在流式模式下可能不工作（funasr 1.3.1 已知问题）")
                logger.info("提示：VAD 会在模型加载时启用，但在流式处理时会被跳过")
            
            # 添加标点恢复
            if self.use_punc:
                punc_model_dir = "./models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
                if os.path.exists(punc_model_dir):
                    model_kwargs["punc_model"] = punc_model_dir
                else:
                    model_kwargs["punc_model"] = "ct-punc"
                logger.info("启用标点恢复")
            
            self.model = AutoModel(**model_kwargs)
            
            logger.info(f"✅ Paraformer-streaming 模型加载完成，设备: {self.device}")
            logger.info(f"块大小配置: {self.chunk_size} = {self.chunk_size[1]*60}ms ≈ {self.chunk_size[1]*60/1000:.1f}秒/块")

    @timeit
    def transcript(self, file_path: str) -> TranscriptResult:
        """
        使用 Paraformer-streaming 进行语音转写（流式模式）
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            TranscriptResult: 转写结果
        """
        import soundfile
        start_time = time.perf_counter()
        
        try:
            # 延迟加载模型
            self._load_model()
            
            logger.info(f"🎤 开始流式转写: {file_path}")
            logger.info(f"使用模型: Paraformer-streaming")
            
            # 读取音频
            speech, sample_rate = soundfile.read(file_path)
            
            # 检查音频格式
            logger.info(f"音频格式: shape={speech.shape}, sample_rate={sample_rate}Hz, dtype={speech.dtype}")
            
            # 如果是立体声，转换为单声道
            if len(speech.shape) > 1:
                logger.info(f"检测到 {speech.shape[1]} 声道音频，转换为单声道")
                speech = speech.mean(axis=1)
                logger.info(f"转换后: shape={speech.shape}")
            
            # 确保是 1 维数组
            if len(speech.shape) != 1:
                raise ValueError(f"音频数据维度错误: {speech.shape}，期望 1 维")
            
            audio_duration = len(speech) / sample_rate
            
            # 检查并调整采样率
            if sample_rate != 16000:
                logger.warning(f"⚠️ 音频采样率为 {sample_rate}Hz，需要重采样到 16000Hz")
                # 使用 soundfile 重采样（简单线性插值）
                import numpy as np
                target_length = int(len(speech) * 16000 / sample_rate)
                speech = np.interp(
                    np.linspace(0, len(speech) - 1, target_length),
                    np.arange(len(speech)),
                    speech
                ).astype(np.float32)
                sample_rate = 16000
                audio_duration = len(speech) / sample_rate
                logger.info(f"✅ 重采样完成: 新长度={len(speech)}, 时长={audio_duration:.2f}秒")
            
            # 计算块大小（基于 16kHz）
            chunk_stride = self.chunk_size[1] * 960  # 10 * 960 = 9600 采样点 = 0.6秒
            total_chunk_num = int(len(speech) / chunk_stride + 1)
            
            logger.info(f"音频时长: {audio_duration:.2f}秒，分为 {total_chunk_num} 个块")
            logger.info(f"每个块: {chunk_stride} 采样点 = {chunk_stride/16000:.2f}秒 @ 16kHz")
            logger.info(f"预计处理时间: {total_chunk_num * 0.4:.1f}秒（假设每块 0.4 秒）")
            
            # 流式处理
            cache = {}
            segments = []
            full_text_parts = []
            
            for i in range(total_chunk_num):
                speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
                is_final = (i == total_chunk_num - 1)
                
                # 确保音频块不为空
                if len(speech_chunk) == 0:
                    logger.warning(f"块 {i} 为空，跳过")
                    continue
                
                # 调试第一个块
                if i == 0:
                    logger.info(f"第一个块: shape={speech_chunk.shape}, dtype={speech_chunk.dtype}, len={len(speech_chunk)}")
                
                try:
                    res = self.model.generate(
                        input=speech_chunk,
                        cache=cache,
                        is_final=is_final,
                        chunk_size=self.chunk_size,
                        encoder_chunk_look_back=self.encoder_chunk_look_back,
                        decoder_chunk_look_back=self.decoder_chunk_look_back,
                    )
                    
                    if res and len(res) > 0:
                        text = res[0].get("text", "").strip()
                        if text:
                            full_text_parts.append(text)
                            
                            # 创建分段
                            start_sec = i * chunk_stride / sample_rate
                            end_sec = min((i + 1) * chunk_stride / sample_rate, audio_duration)
                            
                            segments.append(TranscriptSegment(
                                start=start_sec,
                                end=end_sec,
                                text=text
                            ))
                
                except Exception as e:
                    logger.error(f"处理块 {i} 时出错: {e}")
                    if i == 0:
                        # 第一个块就失败，说明有严重问题
                        raise
                    # 其他块失败，继续处理
                    continue
            
            # 合并完整文本
            full_text = "".join(full_text_parts)
            
            if not full_text:
                raise ValueError("转写结果为空，所有块都没有识别出文本")
            
            # 检测语言
            detected_language = "zh"
            
            transcript_result = TranscriptResult(
                language=detected_language,
                full_text=full_text,
                segments=segments,
                raw={"model": "paraformer-streaming", "chunks": total_chunk_num, "valid_chunks": len(segments)}
            )
            
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"✅ 流式转写完成，耗时 {elapsed_time:.2f} 秒")
            logger.info(f"识别语言: {detected_language}, 文本长度: {len(full_text)}, 有效片段: {len(segments)}/{total_chunk_num}")
            logger.info(f"RTF: {elapsed_time / audio_duration:.3f}")
            
            return transcript_result
            
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(f"❌ Paraformer-streaming 转写失败，耗时 {elapsed_time:.2f} 秒: {e}")
            raise

    def on_finish(self, video_path: str, result: TranscriptResult) -> None:
        """转写完成回调"""
        logger.info(f"Paraformer-streaming 转写完成: {video_path}")
