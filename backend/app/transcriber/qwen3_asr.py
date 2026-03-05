"""
Qwen3-ASR-1.7B 语音识别转写器

特点:
    - 支持流式/离线统一推理
    - 1.7B 参数，SOTA 性能
    - 支持 52 种语言和方言（30 种语言 + 22 种中文方言）
    - 支持时间戳预测（需要 ForcedAligner）
    - 自动检测本地/远程模式
"""

import os
import time
import threading
import base64
import requests
from typing import Optional, List

# 强制使用 ModelScope（国内更快，阿里的模型）
os.environ['MODELSCOPE_CACHE'] = os.path.expanduser('~/.cache/modelscope')
os.environ['MODELSCOPE_MODULES_CACHE'] = os.path.expanduser('~/.cache/modelscope/hub')
# 禁用 Hugging Face
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 如果必须用 HF，使用镜像
os.environ['TRANSFORMERS_OFFLINE'] = '0'

from app.decorators.timeit import timeit
from app.models.transcriber_model import TranscriptResult, TranscriptSegment
from app.transcriber.base import Transcriber
from app.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)


class Qwen3ASRTranscriber(Transcriber):
    """
    Qwen3-ASR-1.7B 语音识别转写器
    
    特点:
        - 1.7B 参数，开源 ASR 模型中的 SOTA
        - 支持 30 种语言 + 22 种中文方言
        - 支持流式和离线推理
        - 可选时间戳预测
        - 基于 vLLM 后端，高性能
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        backend: str = "vllm",
        use_forced_aligner: bool = False,
        gpu_memory_utilization: float = 0.7,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 4096,
        device: str = "cuda",
        remote_url: str = None,
        timeout: int = 300
    ):
        """
        初始化 Qwen3-ASR 转写器
        
        自动检测模式：
        - 如果配置了 remote_url 或环境变量 QWEN3_ASR_REMOTE_URL，使用远程模式
        - 否则使用本地模式（需要 vLLM 支持）
        
        Args:
            model_name: 模型名称
            backend: 后端类型（本地模式）
            use_forced_aligner: 是否使用强制对齐器
            gpu_memory_utilization: GPU 内存利用率（本地模式）
            max_inference_batch_size: 最大批处理大小
            max_new_tokens: 最大生成 token 数
            device: 运行设备（本地模式）
            remote_url: 远程服务地址（远程模式）
            timeout: 请求超时时间（远程模式）
        """
        # 检查是否使用远程模式
        self.remote_url = remote_url or os.getenv("QWEN3_ASR_REMOTE_URL")
        
        if self.remote_url:
            # 远程模式
            self.mode = "remote"
            self.remote_url = self.remote_url.rstrip('/')
            self.timeout = timeout
            self.use_forced_aligner = use_forced_aligner
            
            logger.info("=" * 80)
            logger.info("Qwen3-ASR 转写器初始化（远程模式）")
            logger.info(f"远程服务地址: {self.remote_url}")
            logger.info("=" * 80)
            
            # 检查远程连接
            self._check_remote_connection()
        else:
            # 本地模式
            self.mode = "local"
            self.model_name = model_name
            self.backend = backend
            self.use_forced_aligner = use_forced_aligner
            self.gpu_memory_utilization = gpu_memory_utilization
            self.max_inference_batch_size = max_inference_batch_size
            self.max_new_tokens = max_new_tokens
            
            # 自动检测设备
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA 不可用，将使用 CPU 运行")
                self.device = "cpu"
            else:
                self.device = device
            
            self.model = None
            self._model_lock = threading.Lock()
            
            logger.info("=" * 80)
            logger.info("Qwen3-ASR 转写器初始化（本地模式）")
            logger.info(f"模型: {self.model_name}")
            logger.info(f"后端: {self.backend}")
            logger.info(f"设备: {self.device}")
            logger.info("=" * 80)
            
            # 检查依赖
            self._check_dependencies()

    def _check_remote_connection(self):
        """检查与远程服务的连接"""
        try:
            response = requests.get(
                f"{self.remote_url}/health",
                timeout=5
            )
            
            if response.status_code == 200:
                info = response.json()
                logger.info(f"✅ 远程服务连接成功")
                logger.info(f"  模型: {info.get('model')}")
                logger.info(f"  后端: {info.get('backend')}")
                logger.info(f"  设备: {info.get('device')}")
                if info.get('cuda_available'):
                    logger.info(f"  GPU: {info.get('gpu_name')}")
                logger.info("=" * 80)
            else:
                logger.warning(f"⚠️ 远程服务响应异常: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ 无法连接到远程服务: {self.remote_url}")
            logger.error("请确保:")
            logger.error("  1. WSL2 已启动")
            logger.error("  2. 服务正在运行: python qwen3_asr_server.py")
            logger.error("  3. 检查 QWEN3_ASR_REMOTE_URL 配置")
            raise ConnectionError(f"无法连接到远程服务: {self.remote_url}")
        except Exception as e:
            logger.error(f"❌ 连接检查失败: {e}")
            raise

    def _check_dependencies(self):
        """检查必要的依赖（本地模式）"""
        try:
            import qwen_asr
            logger.info(f"qwen-asr 已安装")
        except ImportError:
            raise ImportError(
                "请安装 qwen-asr 库:\n"
                "pip install -U qwen-asr[vllm]  # 支持流式和高性能\n"
                "或\n"
                "pip install -U qwen-asr  # 仅基础功能"
            )
        
        if self.backend == "vllm":
            try:
                import vllm
                logger.info(f"vLLM 已安装")
            except ImportError:
                logger.warning("vLLM 未安装，将回退到 transformers 后端")
                self.backend = "transformers"
        
        try:
            import torch
            if self.device == "cuda":
                if torch.cuda.is_available():
                    logger.info(f"CUDA 可用，设备: {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning("CUDA 不可用，将使用 CPU")
                    self.device = "cpu"
        except ImportError:
            raise ImportError("请安装 PyTorch")

    def _load_model(self):
        """延迟加载模型（线程安全）"""
        if self.model is None:
            with self._model_lock:
                if self.model is None:
                    from qwen_asr import Qwen3ASRModel
                    import torch
                    
                    logger.info("=" * 80)
                    logger.info(f"正在加载 {self.model_name} 模型...")
                    logger.info(f"后端: {self.backend}")
                    logger.info(f"设备: {self.device}")
                    logger.info("📦 使用 ModelScope 下载模型（国内更快）")
                    logger.info(f"缓存目录: {os.environ['MODELSCOPE_CACHE']}")
                    logger.info("=" * 80)
                    
                    # 根据后端选择加载方式
                    if self.backend == "vllm":
                        # vLLM 后端（推荐）
                        model_kwargs = {
                            "model": self.model_name,
                            "gpu_memory_utilization": self.gpu_memory_utilization,
                            "max_inference_batch_size": self.max_inference_batch_size,
                            "max_new_tokens": self.max_new_tokens,
                        }
                        
                        # 添加强制对齐器
                        if self.use_forced_aligner:
                            model_kwargs["forced_aligner"] = "Qwen/Qwen3-ForcedAligner-0.6B"
                            model_kwargs["forced_aligner_kwargs"] = {
                                "dtype": torch.bfloat16,
                                "device_map": self.device,
                            }
                            logger.info("启用强制对齐器（时间戳预测）")
                        
                        self.model = Qwen3ASRModel.LLM(**model_kwargs)
                    
                    else:
                        # Transformers 后端
                        model_kwargs = {
                            "dtype": torch.bfloat16,
                            "device_map": self.device,
                            "max_inference_batch_size": self.max_inference_batch_size,
                            "max_new_tokens": self.max_new_tokens,
                        }
                        
                        # 添加强制对齐器
                        if self.use_forced_aligner:
                            model_kwargs["forced_aligner"] = "Qwen/Qwen3-ForcedAligner-0.6B"
                            model_kwargs["forced_aligner_kwargs"] = {
                                "dtype": torch.bfloat16,
                                "device_map": self.device,
                            }
                            logger.info("启用强制对齐器（时间戳预测）")
                        
                        self.model = Qwen3ASRModel.from_pretrained(
                            self.model_name,
                            **model_kwargs
                        )
                    
                    logger.info(f"✅ {self.model_name} 模型加载完成")
                    logger.info(f"模型缓存位置: ~/.cache/modelscope/hub/")
                    logger.info("=" * 80)

    @timeit
    def transcript(self, file_path: str) -> TranscriptResult:
        """
        使用 Qwen3-ASR 进行语音转写
        
        自动根据模式选择本地或远程转写
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            TranscriptResult: 转写结果
        """
        if self.mode == "remote":
            return self._transcript_remote(file_path)
        else:
            return self._transcript_local(file_path)
    
    def _transcript_remote(self, file_path: str) -> TranscriptResult:
        """远程转写"""
        start_time = time.perf_counter()
        
        try:
            logger.info(f"🎤 开始远程转写: {file_path}")
            logger.info(f"远程服务: {self.remote_url}")
            
            # 读取音频文件
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            # 编码为 base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 构建请求
            payload = {
                "audio": audio_base64,
                "language": None,
                "return_timestamps": self.use_forced_aligner
            }
            
            # 发送请求
            response = requests.post(
                f"{self.remote_url}/transcribe",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_msg = response.json().get('error', 'Unknown error')
                raise Exception(f"远程服务返回错误: {error_msg}")
            
            # 解析响应
            result_data = response.json()
            
            detected_language = result_data.get('language', 'unknown')
            full_text = result_data.get('text', '')
            processing_time = result_data.get('processing_time', 0)
            
            if not full_text:
                raise ValueError("转写结果为空")
            
            # 处理时间戳
            segments = []
            if self.use_forced_aligner and 'timestamps' in result_data:
                for ts in result_data['timestamps']:
                    segments.append(TranscriptSegment(
                        start=ts['start'],
                        end=ts['end'],
                        text=ts['text']
                    ))
            else:
                segments.append(TranscriptSegment(
                    start=0.0,
                    end=0.0,
                    text=full_text
                ))
            
            # 构建结果
            transcript_result = TranscriptResult(
                language=detected_language,
                full_text=full_text,
                segments=segments,
                raw={
                    "model": "Qwen3-ASR-1.7B (Remote)",
                    "mode": "remote",
                    "remote_url": self.remote_url,
                    "remote_processing_time": processing_time
                }
            )
            
            elapsed_time = time.perf_counter() - start_time
            
            logger.info(f"✅ 远程转写完成，总耗时 {elapsed_time:.2f} 秒")
            logger.info(f"  远程处理: {processing_time:.2f} 秒")
            logger.info(f"  网络传输: {elapsed_time - processing_time:.2f} 秒")
            logger.info(f"  识别语言: {detected_language}")
            logger.info(f"  文本长度: {len(full_text)}")
            
            return transcript_result
            
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(f"❌ 远程转写失败，耗时 {elapsed_time:.2f} 秒: {e}")
            raise
    
    def _transcript_local(self, file_path: str) -> TranscriptResult:
        """本地转写"""
        start_time = time.perf_counter()
        
        try:
            # 延迟加载模型
            self._load_model()
            
            logger.info(f"🎤 开始转写: {file_path}")
            logger.info(f"使用模型: {self.model_name}")
            
            # 获取音频时长（用于计算 RTF）
            audio_duration = self._get_audio_duration(file_path)
            logger.info(f"音频时长: {audio_duration:.2f}秒")
            
            # 转写音频
            results = self.model.transcribe(
                audio=file_path,
                language=None,  # 自动检测语言
                return_time_stamps=self.use_forced_aligner,  # 是否返回时间戳
            )
            
            if not results or len(results) == 0:
                raise ValueError("转写结果为空")
            
            result = results[0]
            
            # 提取语言和文本
            detected_language = result.language if hasattr(result, 'language') else "unknown"
            full_text = result.text if hasattr(result, 'text') else ""
            
            if not full_text:
                raise ValueError("转写文本为空")
            
            # 处理时间戳
            segments = []
            if self.use_forced_aligner and hasattr(result, 'time_stamps') and result.time_stamps:
                # 使用强制对齐器的时间戳
                for ts in result.time_stamps:
                    if hasattr(ts, 'text') and hasattr(ts, 'start_time') and hasattr(ts, 'end_time'):
                        segments.append(TranscriptSegment(
                            start=ts.start_time,
                            end=ts.end_time,
                            text=ts.text
                        ))
                logger.info(f"生成了 {len(segments)} 个时间戳片段")
            else:
                # 没有时间戳，创建单个片段
                segments.append(TranscriptSegment(
                    start=0.0,
                    end=audio_duration,
                    text=full_text
                ))
            
            # 构建结果
            transcript_result = TranscriptResult(
                language=detected_language,
                full_text=full_text,
                segments=segments,
                raw={
                    "model": self.model_name,
                    "mode": "local",
                    "backend": self.backend,
                    "has_timestamps": self.use_forced_aligner
                }
            )
            
            elapsed_time = time.perf_counter() - start_time
            rtf = elapsed_time / audio_duration if audio_duration > 0 else 0
            
            logger.info(f"✅ 转写完成，耗时 {elapsed_time:.2f} 秒")
            logger.info(f"识别语言: {detected_language}, 文本长度: {len(full_text)}, 片段数: {len(segments)}")
            logger.info(f"RTF (实时率): {rtf:.3f}")
            
            return transcript_result
            
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(f"❌ Qwen3-ASR 转写失败，耗时 {elapsed_time:.2f} 秒: {e}")
            raise

    def _get_audio_duration(self, file_path: str) -> float:
        """获取音频时长"""
        try:
            import soundfile
            info = soundfile.info(file_path)
            return info.duration
        except Exception as e:
            logger.warning(f"无法获取音频时长: {e}，使用默认值 0")
            return 0.0

    def on_finish(self, video_path: str, result: TranscriptResult) -> None:
        """转写完成回调"""
        logger.info(f"Qwen3-ASR 转写完成: {video_path}")
