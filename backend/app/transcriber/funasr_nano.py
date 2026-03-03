import os
import sys
from typing import Optional
import threading

from app.decorators.timeit import timeit
from app.models.transcriber_model import TranscriptResult, TranscriptSegment
from app.transcriber.base import Transcriber
from app.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)


class FunASRNanoTranscriber(Transcriber):
    """
    Fun-ASR-Nano 语音识别转写器
    
    特点:
        - 支持 31 种语言，包括中文（7种方言 + 26种地区口音）
        - 支持低延迟实时转写
        - 800M 参数，模型大小约 1.6-3.2 GB
        - 阿里巴巴达摩院 2025年12月最新发布
        - 支持流式和非流式转写
    
    注意：
        Fun-ASR-Nano 需要使用 model.py 中的 FunASRNano 类
        该类需要从模型目录加载
    """

    def __init__(self, model_size: str = "2512", device: str = "cuda"):
        """
        初始化 Fun-ASR-Nano 转写器
        
        Args:
            model_size: 模型版本，默认 "2512" (2025年12月版本)
            device: 运行设备，"cuda" 或 "cpu"，默认自动检测
        """
        self.model_size = model_size
        
        # 自动检测可用设备
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，将使用 CPU 运行")
            logger.info("提示：AMD GPU 需要安装 PyTorch ROCm 版本才能使用 GPU 加速")
            self.device = "cpu"
        else:
            self.device = device if device else "cpu"
        
        self.model = None
        self.model_dir = None
        self._model_lock = threading.Lock()  # 线程锁，保护模型加载
        
        # 检查必要的依赖
        try:
            import funasr
            logger.info(f"FunASR 版本: {funasr.__version__}")
        except ImportError as e:
            raise ImportError(
                f"请安装必要的依赖: {e}\n"
                "pip install funasr torch torchaudio"
            )

    def _load_model(self):
        """延迟加载模型（线程安全）"""
        # 双重检查锁定模式
        if self.model is None:
            with self._model_lock:
                # 再次检查，避免多个线程重复加载
                if self.model is None:
                    from funasr import AutoModel
                    
                    logger.info(f"正在加载 Fun-ASR-Nano-{self.model_size} 模型...")
                    
                    # 优先使用本地已下载的模型
                    local_model_dir = f"./models/FunAudioLLM/Fun-ASR-Nano-{self.model_size}"
                    
                    if os.path.exists(local_model_dir):
                        model_dir = local_model_dir
                        logger.info(f"使用本地模型: {model_dir}")
                        # remote_code 是相对于当前工作目录的路径
                        # 当前工作目录是 backend/，所以路径是相对于 backend/
                        remote_code_path = f"./models/FunAudioLLM/Fun-ASR-Nano-{self.model_size}/model.py"
                    else:
                        # 如果本地不存在，从 ModelScope 下载
                        model_dir = f"FunAudioLLM/Fun-ASR-Nano-{self.model_size}"
                        logger.info(f"本地模型不存在，从 ModelScope 下载: {model_dir}")
                        remote_code_path = "./model.py"  # ModelScope 会自动下载到模型目录
                    
                    try:
                        # Fun-ASR-Nano 需要使用 trust_remote_code=True 和 remote_code 参数
                        # 参考 demo1.py 的用法
                        # 注意：不使用 VAD，因为 funasr 1.3.1 的 VAD 有 bug (KeyError: 0)
                        self.model = AutoModel(
                            model=model_dir,
                            trust_remote_code=True,
                            remote_code=remote_code_path,  # 相对于当前工作目录的路径
                            device=self.device,
                            hub="ms",  # 使用 ModelScope
                            disable_update=True,  # 禁用版本检查，加快加载速度
                        )
                        logger.info(f"✅ Fun-ASR-Nano 模型加载完成（无 VAD），设备: {self.device}")
                    except Exception as e:
                        logger.error(f"加载失败: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        raise Exception(
                            f"Fun-ASR-Nano 模型加载失败: {e}\n"
                            f"请确保模型已下载或 ModelScope 可访问"
                        )

    @timeit
    def transcript(self, file_path: str) -> TranscriptResult:
        """
        使用 Fun-ASR-Nano 进行语音转写
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            TranscriptResult: 转写结果
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # 延迟加载模型
            self._load_model()
            
            logger.info(f"🎤 开始音频转写: {file_path}")
            logger.info(f"使用模型: Fun-ASR-Nano-{self.model_size}")
            
            # 调用模型进行转写
            # 根据 demo1.py 示例
            res = self.model.generate(
                input=[file_path],
                cache={},
                batch_size=1,
                language="中文",  # 指定中文
                itn=True,  # 启用逆文本归一化
            )
            
            if not res or len(res) == 0:
                raise Exception("Fun-ASR-Nano 返回结果为空")
            
            result = res[0]
            logger.info(f"Fun-ASR-Nano 返回结果键: {result.keys()}")
            
            # 使用 ctc_text 字段，这个更准确（没有重复问题）
            full_text = result.get("ctc_text", result.get("text", "")).strip()
            
            # Fun-ASR-Nano 的时间戳是字符级的，太细了，不适合分段
            # 直接使用完整文本，不分段
            segments = [TranscriptSegment(
                start=0.0,
                end=0.0,
                text=full_text,
            )]
            
            logger.info(f"使用 ctc_text 字段，不进行时间戳分段")
            
            # 检测语言
            detected_language = result.get("language", "zh")
            
            transcript_result = TranscriptResult(
                language=detected_language,
                full_text=full_text,
                segments=segments,
                raw=result,
            )
            
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"✅ 音频转写完成，耗时 {elapsed_time:.2f} 秒")
            logger.info(f"识别语言: {detected_language}, 文本长度: {len(full_text)}, 片段数: {len(segments)}")
            
            return transcript_result
            
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(f"❌ Fun-ASR-Nano 转写失败，耗时 {elapsed_time:.2f} 秒: {e}")
            raise

    def on_finish(self, video_path: str, result: TranscriptResult) -> None:
        """转写完成回调"""
        logger.info(f"Fun-ASR-Nano 转写完成: {video_path}")
