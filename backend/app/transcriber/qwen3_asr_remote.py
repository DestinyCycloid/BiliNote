"""
Qwen3-ASR 远程转写器（Windows 客户端）

通过 HTTP API 调用 Linux 虚拟机上的 Qwen3-ASR 服务
"""

import os
import time
import base64
import requests
from typing import Optional

from app.decorators.timeit import timeit
from app.models.transcriber_model import TranscriptResult, TranscriptSegment
from app.transcriber.base import Transcriber
from app.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)


class Qwen3ASRRemoteTranscriber(Transcriber):
    """
    Qwen3-ASR 远程转写器
    
    通过 HTTP API 调用远程 Linux 服务器上的 Qwen3-ASR 服务
    """

    def __init__(
        self,
        remote_url: str = None,
        timeout: int = 300,
        use_timestamps: bool = False
    ):
        """
        初始化远程转写器
        
        Args:
            remote_url: 远程服务地址，例如 http://192.168.1.100:8765
            timeout: 请求超时时间（秒）
            use_timestamps: 是否返回时间戳
        """
        self.remote_url = remote_url or os.getenv("QWEN3_ASR_REMOTE_URL", "http://localhost:8765")
        self.timeout = timeout
        self.use_timestamps = use_timestamps
        
        # 移除末尾的斜杠
        self.remote_url = self.remote_url.rstrip('/')
        
        logger.info(f"Qwen3-ASR 远程转写器初始化")
        logger.info(f"远程服务地址: {self.remote_url}")
        
        # 检查连接
        self._check_connection()

    def _check_connection(self):
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
            else:
                logger.warning(f"⚠️ 远程服务响应异常: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ 无法连接到远程服务: {self.remote_url}")
            logger.error("请确保:")
            logger.error("  1. Linux 虚拟机已启动")
            logger.error("  2. Qwen3-ASR 服务正在运行")
            logger.error("  3. 网络连接正常")
            logger.error("  4. 防火墙允许访问")
            raise ConnectionError(f"无法连接到远程服务: {self.remote_url}")
        except Exception as e:
            logger.error(f"❌ 连接检查失败: {e}")
            raise

    @timeit
    def transcript(self, file_path: str) -> TranscriptResult:
        """
        使用远程 Qwen3-ASR 服务进行语音转写
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            TranscriptResult: 转写结果
        """
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
                "language": None,  # 自动检测
                "return_timestamps": self.use_timestamps
            }
            
            # 发送请求
            logger.info(f"发送转写请求...")
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
            if self.use_timestamps and 'timestamps' in result_data:
                for ts in result_data['timestamps']:
                    segments.append(TranscriptSegment(
                        start=ts['start'],
                        end=ts['end'],
                        text=ts['text']
                    ))
                logger.info(f"生成了 {len(segments)} 个时间戳片段")
            else:
                # 没有时间戳，创建单个片段
                segments.append(TranscriptSegment(
                    start=0.0,
                    end=0.0,  # 未知时长
                    text=full_text
                ))
            
            # 构建结果
            transcript_result = TranscriptResult(
                language=detected_language,
                full_text=full_text,
                segments=segments,
                raw={
                    "model": "Qwen3-ASR-1.7B (Remote)",
                    "remote_url": self.remote_url,
                    "remote_processing_time": processing_time
                }
            )
            
            elapsed_time = time.perf_counter() - start_time
            
            logger.info(f"✅ 远程转写完成，总耗时 {elapsed_time:.2f} 秒")
            logger.info(f"  远程处理时间: {processing_time:.2f} 秒")
            logger.info(f"  网络传输时间: {elapsed_time - processing_time:.2f} 秒")
            logger.info(f"  识别语言: {detected_language}")
            logger.info(f"  文本长度: {len(full_text)}")
            logger.info(f"  片段数: {len(segments)}")
            
            return transcript_result
            
        except requests.exceptions.Timeout:
            elapsed_time = time.perf_counter() - start_time
            logger.error(f"❌ 请求超时，耗时 {elapsed_time:.2f} 秒")
            logger.error(f"超时时间: {self.timeout} 秒")
            logger.error("建议: 增加 timeout 参数或检查网络连接")
            raise
            
        except requests.exceptions.ConnectionError:
            elapsed_time = time.perf_counter() - start_time
            logger.error(f"❌ 连接失败，耗时 {elapsed_time:.2f} 秒")
            logger.error("请检查:")
            logger.error("  1. 远程服务是否正在运行")
            logger.error("  2. 网络连接是否正常")
            logger.error("  3. 防火墙设置")
            raise
            
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(f"❌ 远程转写失败，耗时 {elapsed_time:.2f} 秒: {e}")
            raise

    def on_finish(self, video_path: str, result: TranscriptResult) -> None:
        """转写完成回调"""
        logger.info(f"Qwen3-ASR 远程转写完成: {video_path}")
