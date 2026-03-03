"""
实时音频捕获 + Faster-Whisper 流式转写测试

使用项目现有的 Faster-Whisper 模型进行流式转写
虽然 Faster-Whisper 不是真正的流式，但可以通过 VAD 分段模拟流式效果

依赖安装：
pip install pyaudiowpatch numpy faster-whisper

使用方法：
python test_realtime_audio_faster_whisper.py
"""

import sys
import os
import time
import threading
import queue
import numpy as np
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.logger import get_logger

logger = get_logger(__name__)


class RealtimeAudioFasterWhisper:
    """实时音频流式转写（Faster-Whisper + VAD）"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 3,  # 建议 2-5 秒
        device: str = "cuda"
    ):
        """
        初始化实时音频流式转写
        
        Args:
            sample_rate: 采样率（Hz），Whisper 需要 16000Hz
            chunk_duration: 每个音频块的时长（秒）
            device: 运行设备（cuda 或 cpu）
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.device = device
        
        # 音频队列
        self.audio_queue = queue.Queue()
        
        # 控制标志
        self.is_recording = False
        self.is_processing = False
        
        # 模型
        self.model = None
        
        # PyAudio 对象
        self.pyaudio = None
        self.stream = None
        
        # 检查依赖
        self._check_dependencies()
        
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
            import faster_whisper
            logger.info(f"✅ faster-whisper 已安装")
        except ImportError:
            raise ImportError(
                "请安装 faster-whisper 库:\n"
                "pip install faster-whisper"
            )
        
        try:
            import torch
            if self.device == "cuda":
                if torch.cuda.is_available():
                    logger.info(f"✅ CUDA 可用，设备: {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning("⚠️ CUDA 不可用，将使用 CPU")
                    self.device = "cpu"
            else:
                logger.info(f"使用 CPU 运行")
        except ImportError:
            raise ImportError("请安装 PyTorch")
    
    def _load_model(self):
        """加载 Faster-Whisper large-v3 模型"""
        if self.model is not None:
            return
        
        from faster_whisper import WhisperModel
        
        logger.info("=" * 80)
        logger.info("正在加载 Faster-Whisper large-v3 模型...")
        logger.info("=" * 80)
        
        # 只使用 large-v3 模型
        model_path = "./models/whisper/whisper-large-v3"
        
        # 检查模型是否存在
        if not os.path.exists(model_path):
            logger.error(f"❌ large-v3 模型不存在: {model_path}")
            logger.error("")
            logger.error("请先下载模型：")
            logger.error("  python download_faster_whisper_large.py")
            logger.error("")
            raise FileNotFoundError(f"large-v3 模型不存在: {model_path}")
        
        logger.info(f"模型路径: {model_path}")
        logger.info("参数量: 1550M, 模型大小: ~3 GB")
        
        try:
            # 加载模型
            self.model = WhisperModel(
                model_path,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8",
            )
            
            logger.info(f"✅ Faster-Whisper 模型加载完成，设备: {self.device}")
            logger.info(f"块大小: {self.chunk_size} 采样点 = {self.chunk_duration:.2f} 秒")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def _list_audio_devices(self):
        """列出所有音频设备"""
        p = self.pyaudio.PyAudio()
        
        logger.info("\n" + "=" * 80)
        logger.info("可用的音频设备：")
        logger.info("=" * 80)
        
        loopback_device = None
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            
            if info.get("isLoopbackDevice"):
                logger.info(f"\n🔊 找到 Loopback 设备（系统音频捕获）:")
                logger.info(f"  设备 ID: {i}")
                logger.info(f"  设备名称: {info['name']}")
                logger.info(f"  采样率: {int(info['defaultSampleRate'])}Hz")
                logger.info(f"  输入声道: {info['maxInputChannels']}")
                loopback_device = info
        
        if not loopback_device:
            try:
                default_speakers = p.get_default_wasapi_loopback()
                logger.info(f"\n✅ 找到默认扬声器:")
                logger.info(f"  设备 ID: {default_speakers['index']}")
                logger.info(f"  设备名称: {default_speakers['name']}")
                logger.info(f"  采样率: {int(default_speakers['defaultSampleRate'])}Hz")
                loopback_device = default_speakers
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
                logger.info(f"设备采样率: {device_sample_rate}Hz")
                logger.info(f"目标采样率: {self.sample_rate}Hz")
                
                device_chunk_size = int(device_sample_rate * self.chunk_duration)
                
                logger.info(f"开始捕获系统音频")
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
                        
                        if wasapi_info['maxInputChannels'] > 1:
                            audio_array = audio_array.reshape(-1, wasapi_info['maxInputChannels'])
                            audio_array = audio_array.mean(axis=1)
                        
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
        """音频处理线程（使用 VAD 分段模拟流式）"""
        try:
            chunk_index = 0
            all_sentences = []
            current_line = []
            current_line_start_time = None
            max_display_lines = 15
            
            self._clear_screen()
            print("=" * 80)
            print("📺 Faster-Whisper 流式转写（VAD 分段）")
            print("=" * 80)
            print()
            
            while self.is_processing:
                try:
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                    
                    audio_energy = np.abs(audio_chunk).mean()
                    if audio_energy < 0.001:
                        chunk_index += 1
                        continue
                    
                    start_time = time.perf_counter()
                    
                    # Faster-Whisper 转写（使用 VAD 分段）
                    segments, info = self.model.transcribe(
                        audio_chunk,
                        language="zh",  # 指定中文
                        vad_filter=True,  # 启用 VAD
                        vad_parameters=dict(
                            min_silence_duration_ms=500,  # 最小静音时长
                            speech_pad_ms=400,  # 语音填充
                        ),
                    )
                    
                    elapsed = time.perf_counter() - start_time
                    
                    # 处理分段结果
                    for segment in segments:
                        text = segment.text.strip()
                        if text:
                            timestamp = time.strftime("%H:%M:%S")
                            
                            if not current_line:
                                current_line_start_time = timestamp
                            
                            current_line.append(text)
                            current_text = "".join(current_line)
                            
                            should_newline = (
                                len(current_text) >= 40 or
                                any(p in text for p in ['。', '！', '？'])
                            )
                            
                            if should_newline:
                                all_sentences.append((current_line_start_time, current_text))
                                current_line = []
                                current_line_start_time = None
                            
                            self._refresh_display_with_current(
                                all_sentences,
                                current_line,
                                current_line_start_time,
                                max_display_lines
                            )
                    
                    logger.debug(f"处理性能: {elapsed*1000:.0f}ms, 音频能量: {audio_energy:.4f}")
                    
                    chunk_index += 1
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ 处理块 {chunk_index} 时出错: {e}")
                    continue
            
            # 处理结束
            if current_line:
                current_text = "".join(current_line)
                all_sentences.append((current_line_start_time, current_text))
            
            if len(all_sentences) > 0:
                self._clear_screen()
                print("\n" + "=" * 80)
                print("📝 转写完成！完整内容：")
                print("=" * 80)
                print()
                for timestamp, text in all_sentences:
                    print(f"[{timestamp}] {text}")
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
    
    def _refresh_display_with_current(self, all_sentences, current_line, current_time, max_lines):
        """刷新显示"""
        self._clear_screen()
        
        total_chars = sum(len(t[1]) for t in all_sentences)
        if current_line:
            total_chars += len("".join(current_line))
        
        print("=" * 80)
        print("📺 Faster-Whisper 实时转写")
        print("=" * 80)
        print(f"已识别：{len(all_sentences)} 句，{total_chars} 字")
        print("=" * 80)
        print()
        
        display_sentences = all_sentences[-max_lines:] if len(all_sentences) > max_lines else all_sentences
        
        if len(all_sentences) > max_lines:
            print(f"... (前面还有 {len(all_sentences) - max_lines} 句)")
            print()
        
        for timestamp, text in display_sentences:
            print(f"[{timestamp}] {text}")
        
        if current_line:
            current_text = "".join(current_line)
            print(f"[{current_time}] {current_text} ...")
        
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
        
        self._load_model()
        
        self.is_recording = True
        capture_thread = threading.Thread(
            target=self._audio_capture_thread,
            daemon=True
        )
        capture_thread.start()
        
        self.is_processing = True
        processing_thread = threading.Thread(
            target=self._audio_processing_thread,
            daemon=True
        )
        processing_thread.start()
        
        logger.info("\n" + "=" * 80)
        logger.info("🎤 Faster-Whisper 流式转写已启动！")
        logger.info("=" * 80)
        logger.info("提示：")
        logger.info("  - 使用项目现有的 Faster-Whisper 模型")
        logger.info("  - 通过 VAD 分段模拟流式效果")
        logger.info("  - 延迟约 3-5 秒")
        logger.info("  - 按 Ctrl+C 停止录音")
        if duration:
            logger.info(f"  - 将在 {duration} 秒后自动停止")
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
        logger.info("✅ Faster-Whisper 流式转写已停止")
        logger.info("=" * 80)


def main():
    """主函数"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 18 + "Faster-Whisper 流式转写测试" + " " * 18 + "║")
    print("║" + " " * 20 + "（使用项目现有模型）" + " " * 20 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    streaming = RealtimeAudioFasterWhisper(
        sample_rate=16000,
        chunk_duration=3.0,  # 3 秒一个块
        device="cuda"
    )
    
    streaming.start(duration=None)


if __name__ == "__main__":
    main()
