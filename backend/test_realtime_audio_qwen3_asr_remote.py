"""
实时音频捕获 + Qwen3-ASR 远程转写测试（调用 WSL2 服务）

功能：
1. 实时捕获 Windows 系统音频（浏览器播放的视频、音乐等）
2. 通过 HTTP API 调用 WSL2 上的 Qwen3-ASR 服务进行转写
3. 实时显示字幕效果
4. 支持 52 种语言和方言

架构：
- Windows: 捕获音频 + 显示结果
- WSL2: 运行 Qwen3-ASR + vLLM 服务

依赖安装：
pip install pyaudiowpatch numpy requests soundfile

使用方法：
1. 先在 WSL2 中启动服务: python qwen3_asr_server.py
2. 然后在 Windows 中运行: python test_realtime_audio_qwen3_asr_remote.py

环境变量配置（可选）：
QWEN3_ASR_REMOTE_URL=http://WSL2的IP:8765
"""

import sys
import os
import time
import threading
import queue
import numpy as np
import requests
import base64
import io
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.logger import get_logger

logger = get_logger(__name__)


class RealtimeQwen3ASRRemoteStreaming:
    """实时音频流式转写（Windows 系统音频捕获 + WSL2 远程服务）"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 3.0,  # 每个块的时长（秒）
        remote_url: str = None,
        timeout: int = 30
    ):
        """
        初始化实时音频流式转写
        
        Args:
            sample_rate: 采样率（Hz）
            chunk_duration: 每个音频块的时长（秒），建议 2-5 秒
            remote_url: 远程服务地址，例如 http://172.28.176.123:8765
            timeout: 请求超时时间（秒）
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.remote_url = remote_url or os.getenv("QWEN3_ASR_REMOTE_URL", "http://localhost:8765")
        self.timeout = timeout
        
        # 移除末尾的斜杠
        self.remote_url = self.remote_url.rstrip('/')
        
        # 音频队列
        self.audio_queue = queue.Queue()
        
        # 控制标志
        self.is_recording = False
        self.is_processing = False
        
        # PyAudio 对象
        self.pyaudio = None
        self.stream = None
        
        # 检查依赖
        self._check_dependencies()
        
        # 检查远程服务连接
        self._check_remote_connection()
        
    def _check_dependencies(self):
        """检查必要的依赖"""
        try:
            import pyaudiowpatch as pyaudio
            logger.info(f"✅ pyaudiowpatch 已安装")
            self.pyaudio = pyaudio
        except ImportError:
            raise ImportError(
                "请安装 pyaudiowpatch 库:\n"
                "pip install pyaudiowpatch"
            )
        
        try:
            import soundfile
            logger.info(f"✅ soundfile 已安装")
        except ImportError:
            raise ImportError(
                "请安装 soundfile 库:\n"
                "pip install soundfile"
            )
    
    def _check_remote_connection(self):
        """检查与远程服务的连接"""
        logger.info("=" * 80)
        logger.info("检查远程服务连接...")
        logger.info(f"远程服务地址: {self.remote_url}")
        logger.info("=" * 80)
        
        try:
            response = requests.get(
                f"{self.remote_url}/health",
                timeout=5
            )
            
            if response.status_code == 200:
                info = response.json()
                logger.info(f"✅ 远程服务连接成功！")
                logger.info(f"  模型: {info.get('model')}")
                logger.info(f"  后端: {info.get('backend')}")
                logger.info(f"  设备: {info.get('device')}")
                if info.get('cuda_available'):
                    logger.info(f"  GPU: {info.get('gpu_name')}")
                logger.info("=" * 80)
            else:
                logger.warning(f"⚠️ 远程服务响应异常: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.error("=" * 80)
            logger.error(f"❌ 无法连接到远程服务: {self.remote_url}")
            logger.error("=" * 80)
            logger.error("请确保:")
            logger.error("  1. WSL2 已启动: wsl")
            logger.error("  2. 服务正在运行: python qwen3_asr_server.py")
            logger.error("  3. 获取 WSL2 IP: hostname -I")
            logger.error("  4. 更新 .env 中的 QWEN3_ASR_REMOTE_URL")
            logger.error("=" * 80)
            raise ConnectionError(f"无法连接到远程服务: {self.remote_url}")
        except Exception as e:
            logger.error(f"❌ 连接检查失败: {e}")
            raise
    
    def _list_audio_devices(self):
        """列出所有音频设备"""
        p = self.pyaudio.PyAudio()
        
        logger.info("\n" + "=" * 80)
        logger.info("可用的音频设备：")
        logger.info("=" * 80)
        
        loopback_device = None
        
        # 查找 loopback 设备
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            
            if info.get("isLoopbackDevice"):
                logger.info(f"\n🔊 找到 Loopback 设备（系统音频捕获）:")
                logger.info(f"  设备 ID: {i}")
                logger.info(f"  设备名称: {info['name']}")
                logger.info(f"  采样率: {int(info['defaultSampleRate'])}Hz")
                loopback_device = info
        
        if not loopback_device:
            try:
                loopback_device = p.get_default_wasapi_loopback()
                logger.info(f"\n✅ 找到默认扬声器:")
                logger.info(f"  设备名称: {loopback_device['name']}")
            except Exception as e:
                logger.error(f"❌ 无法获取默认扬声器: {e}")
        
        p.terminate()
        logger.info("=" * 80 + "\n")
        
        return loopback_device
    
    def _audio_capture_thread(self):
        """音频捕获线程"""
        try:
            logger.info("=" * 80)
            logger.info("🔊 使用 WASAPI Loopback 模式捕获系统音频")
            logger.info("=" * 80)
            
            p = self.pyaudio.PyAudio()
            
            try:
                wasapi_info = p.get_default_wasapi_loopback()
                logger.info(f"扬声器设备: {wasapi_info['name']}")
                
                device_sample_rate = int(wasapi_info['defaultSampleRate'])
                device_chunk_size = int(device_sample_rate * self.chunk_duration)
                
                logger.info(f"设备采样率: {device_sample_rate}Hz")
                logger.info(f"目标采样率: {self.sample_rate}Hz")
                logger.info(f"每个块: {self.chunk_duration:.2f} 秒")
                logger.info("=" * 80 + "\n")
                
                self.stream = p.open(
                    format=self.pyaudio.paInt16,
                    channels=wasapi_info['maxInputChannels'],
                    rate=device_sample_rate,
                    input=True,
                    input_device_index=wasapi_info['index'],
                    frames_per_buffer=device_chunk_size,
                )
                
                logger.info("✅ 音频流已打开，开始捕获...\n")
                
                while self.is_recording:
                    try:
                        audio_data = self.stream.read(device_chunk_size, exception_on_overflow=False)
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        audio_array = audio_array.astype(np.float32) / 32768.0
                        
                        # 转换为单声道
                        if wasapi_info['maxInputChannels'] > 1:
                            audio_array = audio_array.reshape(-1, wasapi_info['maxInputChannels'])
                            audio_array = audio_array.mean(axis=1)
                        
                        # 重采样
                        if device_sample_rate != self.sample_rate:
                            target_length = int(len(audio_array) * self.sample_rate / device_sample_rate)
                            audio_array = np.interp(
                                np.linspace(0, len(audio_array) - 1, target_length),
                                np.arange(len(audio_array)),
                                audio_array
                            ).astype(np.float32)
                        
                        self.audio_queue.put(audio_array)
                        
                    except Exception as e:
                        logger.error(f"读取音频数据时出错: {e}")
                        continue
                
            finally:
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                p.terminate()
                logger.info("音频流已关闭")
        
        except Exception as e:
            logger.error(f"❌ 音频捕获失败: {e}")
            self.is_recording = False
    
    def _audio_processing_thread(self):
        """音频处理线程（调用远程服务）"""
        try:
            import soundfile as sf
            
            chunk_index = 0
            all_sentences = []
            current_line = []
            current_line_start_time = None
            max_display_lines = 15
            
            # 初始化显示
            self._clear_screen()
            print("=" * 80)
            print("📺 实时字幕（Qwen3-ASR 远程服务）")
            print("=" * 80)
            print()
            
            while self.is_processing:
                try:
                    # 从队列获取音频块
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                    
                    # 检查音频能量
                    audio_energy = np.abs(audio_chunk).mean()
                    if audio_energy < 0.001:
                        chunk_index += 1
                        continue
                    
                    # 转写
                    start_time = time.perf_counter()
                    
                    try:
                        # 将音频转换为 WAV 格式（内存中）
                        audio_buffer = io.BytesIO()
                        sf.write(audio_buffer, audio_chunk, self.sample_rate, format='WAV')
                        audio_bytes = audio_buffer.getvalue()
                        
                        # 编码为 base64
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        
                        # 发送到远程服务
                        response = requests.post(
                            f"{self.remote_url}/transcribe",
                            json={
                                "audio": audio_base64,
                                "language": None,
                                "return_timestamps": False
                            },
                            timeout=self.timeout
                        )
                        
                        elapsed = time.perf_counter() - start_time
                        
                        if response.status_code == 200:
                            result_data = response.json()
                            text = result_data.get('text', '').strip()
                            detected_lang = result_data.get('language', 'unknown')
                            
                            if text:
                                timestamp = time.strftime("%H:%M:%S")
                                
                                if not current_line:
                                    current_line_start_time = timestamp
                                
                                current_line.append(text)
                                current_text = "".join(current_line)
                                
                                # 判断是否换行
                                should_newline = (
                                    len(current_text) >= 50 or
                                    any(p in text for p in ['。', '！', '？', '.', '!', '?'])
                                )
                                
                                if should_newline:
                                    all_sentences.append((current_line_start_time, current_text, detected_lang))
                                    current_line = []
                                    current_line_start_time = None
                                
                                # 刷新显示
                                self._refresh_display_with_current(
                                    all_sentences,
                                    current_line,
                                    current_line_start_time,
                                    detected_lang,
                                    max_display_lines
                                )
                                
                                # 性能信息
                                if chunk_index % 5 == 0:
                                    logger.debug(f"处理性能: {elapsed*1000:.0f}ms, 语言: {detected_lang}")
                        else:
                            logger.error(f"远程服务返回错误: {response.status_code}")
                    
                    except requests.exceptions.Timeout:
                        logger.error(f"请求超时（块 {chunk_index}）")
                    except Exception as e:
                        logger.error(f"转写块 {chunk_index} 时出错: {e}")
                    
                    chunk_index += 1
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ 处理块 {chunk_index} 时出错: {e}")
                    continue
            
            # 处理结束
            if current_line:
                current_text = "".join(current_line)
                all_sentences.append((current_line_start_time, current_text, "unknown"))
            
            # 显示完整内容
            if len(all_sentences) > 0:
                self._clear_screen()
                print("\n" + "=" * 80)
                print("📝 转写完成！完整内容：")
                print("=" * 80)
                print()
                for timestamp, text, lang in all_sentences:
                    print(f"[{timestamp}] [{lang}] {text}")
                print()
                print("=" * 80)
                print(f"总计：{len(all_sentences)} 句，{sum(len(t[1]) for t in all_sentences)} 字")
                print("=" * 80 + "\n")
        
        except Exception as e:
            logger.error(f"❌ 音频处理失败: {e}")
            self.is_processing = False
    
    def _clear_screen(self):
        """清空屏幕"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _refresh_display_with_current(self, all_sentences, current_line, current_time, current_lang, max_lines):
        """刷新显示"""
        self._clear_screen()
        
        total_chars = sum(len(t[1]) for t in all_sentences)
        if current_line:
            total_chars += len("".join(current_line))
        
        print("=" * 80)
        print("📺 实时字幕（Qwen3-ASR 远程服务）")
        print("=" * 80)
        print(f"已识别：{len(all_sentences)} 句，{total_chars} 字")
        print(f"远程服务：{self.remote_url}")
        print("=" * 80)
        print()
        
        display_sentences = all_sentences[-max_lines:] if len(all_sentences) > max_lines else all_sentences
        
        if len(all_sentences) > max_lines:
            print(f"... (前面还有 {len(all_sentences) - max_lines} 句)")
            print()
        
        for timestamp, text, lang in display_sentences:
            print(f"[{timestamp}] [{lang}] {text}")
        
        if current_line:
            current_text = "".join(current_line)
            print(f"[{current_time}] [{current_lang}] {current_text} ...")
        
        print()
        print("-" * 80)
        print("提示：按 Ctrl+C 停止录音")
        print("-" * 80)
    
    def start(self, duration: Optional[float] = None):
        """开始实时转写"""
        loopback_device = self._list_audio_devices()
        
        if not loopback_device:
            logger.error("❌ 未找到可用的 Loopback 设备")
            return
        
        # 启动录音线程
        self.is_recording = True
        capture_thread = threading.Thread(
            target=self._audio_capture_thread,
            daemon=True
        )
        capture_thread.start()
        
        # 启动处理线程
        self.is_processing = True
        processing_thread = threading.Thread(
            target=self._audio_processing_thread,
            daemon=True
        )
        processing_thread.start()
        
        logger.info("\n" + "=" * 80)
        logger.info("🎤 实时转写已启动！")
        logger.info("=" * 80)
        logger.info("提示：")
        logger.info("  - 播放任何音频，转写结果会实时显示")
        logger.info("  - 支持 52 种语言和方言，自动识别")
        logger.info("  - 按 Ctrl+C 停止录音")
        logger.info("=" * 80 + "\n")
        
        try:
            if duration:
                time.sleep(duration)
                logger.info(f"\n⏰ 录音时长已达 {duration} 秒，停止录音...")
            else:
                while self.is_recording:
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.info("\n\n⏹️  收到停止信号，正在停止...")
        
        finally:
            self.stop()
    
    def stop(self):
        """停止实时转写"""
        self.is_recording = False
        self.is_processing = False
        time.sleep(1)
        logger.info("\n" + "=" * 80)
        logger.info("✅ 实时转写已停止")
        logger.info("=" * 80)


def main():
    """主函数"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "实时音频流式转写测试（远程服务）" + " " * 15 + "║")
    print("║" + " " * 12 + "Qwen3-ASR-1.7B (WSL2) + Windows 捕获" + " " * 12 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # 从环境变量或用户输入获取 WSL2 IP
    remote_url = os.getenv("QWEN3_ASR_REMOTE_URL")
    
    if not remote_url:
        print("=" * 80)
        print("请输入 WSL2 服务地址")
        print("=" * 80)
        print("提示：")
        print("  1. 在 WSL2 中运行: hostname -I")
        print("  2. 获取 IP 地址，例如: 172.28.176.123")
        print("  3. 输入完整地址，例如: http://172.28.176.123:8765")
        print()
        print("或者在 backend/.env 中设置:")
        print("  QWEN3_ASR_REMOTE_URL=http://172.28.176.123:8765")
        print("=" * 80)
        print()
        
        remote_url = input("请输入服务地址 (直接回车使用 http://localhost:8765): ").strip()
        if not remote_url:
            remote_url = "http://localhost:8765"
    
    # 创建实时转写器
    streaming = RealtimeQwen3ASRRemoteStreaming(
        sample_rate=16000,
        chunk_duration=3.0,
        remote_url=remote_url,
        timeout=30
    )
    
    # 开始实时转写
    streaming.start(duration=None)


if __name__ == "__main__":
    main()
