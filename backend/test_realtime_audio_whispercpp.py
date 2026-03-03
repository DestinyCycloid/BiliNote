"""
实时音频捕获 + whisper.cpp 流式转写测试

whisper.cpp 是 OpenAI Whisper 的 C/C++ 实现，支持流式转写

依赖安装：
pip install pyaudiowpatch numpy pywhispercpp

模型下载：
# 方式 1：使用 pywhispercpp 自动下载
from pywhispercpp.model import Model
model = Model('base')  # 会自动下载

# 方式 2：手动下载 ggml 格式模型
# 从 https://huggingface.co/ggerganov/whisper.cpp/tree/main 下载
# 例如：ggml-base.bin

使用方法：
python test_realtime_audio_whispercpp.py
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


class RealtimeAudioWhisperCpp:
    """实时音频流式转写（whisper.cpp）"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 3.0,  # whisper.cpp 建议 3-5 秒
        model_name: str = "base",
        device: str = "cpu"
    ):
        """
        初始化实时音频流式转写
        
        Args:
            sample_rate: 采样率（Hz），Whisper 需要 16000Hz
            chunk_duration: 每个音频块的时长（秒），whisper.cpp 建议 3-5 秒
            model_name: 模型名称（tiny, base, small, medium, large）
            device: 运行设备（cpu 或 cuda）
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.model_name = model_name
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
            import pywhispercpp
            logger.info(f"✅ pywhispercpp 已安装")
        except ImportError:
            raise ImportError(
                "请安装 pywhispercpp 库:\n"
                "pip install pywhispercpp\n\n"
                "注意：这是 whisper.cpp 的 Python 绑定"
            )
    
    def _load_model(self):
        """加载 whisper.cpp 模型"""
        if self.model is not None:
            return
        
        from pywhispercpp.model import Model
        
        logger.info("=" * 80)
        logger.info(f"正在加载 whisper.cpp 模型: {self.model_name}")
        logger.info("=" * 80)
        
        # 使用项目统一的模型目录，但区分 whisper.cpp 和 faster-whisper
        models_dir = "./models/whisper/whisper-cpp"
        model_path = f"{models_dir}/ggml-{self.model_name}.bin"
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logger.info(f"创建 whisper.cpp 模型目录: {models_dir}")
        
        if os.path.exists(model_path):
            logger.info(f"使用本地模型: {model_path}")
        else:
            logger.info(f"本地模型不存在: {model_path}")
            logger.info("提示：首次运行会自动下载模型，请耐心等待...")
            logger.info(f"模型大小参考：")
            logger.info(f"  - tiny: ~75 MB")
            logger.info(f"  - base: ~142 MB")
            logger.info(f"  - small: ~466 MB")
            logger.info(f"  - medium: ~1.5 GB")
            logger.info(f"  - large: ~2.9 GB")
        
        logger.info("=" * 80)
        
        try:
            # pywhispercpp 会自动下载模型
            # 通过设置 models_dir 参数指定模型目录
            self.model = Model(
                self.model_name,
                n_threads=6,
                models_dir=models_dir,  # 指定 whisper.cpp 专用目录
            )
            
            logger.info(f"✅ whisper.cpp 模型加载完成")
            logger.info(f"模型路径: {model_path}")
            logger.info(f"块大小: {self.chunk_size} 采样点 = {self.chunk_duration:.2f} 秒")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            logger.error("提示：")
            logger.error("  1. 确保网络连接正常（首次需要下载模型）")
            logger.error(f"  2. 或手动下载模型到 {models_dir}/")
            logger.error("  3. 模型下载地址：https://huggingface.co/ggerganov/whisper.cpp")
            logger.error("")
            logger.error("注意：whisper.cpp 使用 ggml 格式，与 Faster-Whisper 不兼容")
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
        """音频处理线程（流式转写）"""
        try:
            chunk_index = 0
            all_sentences = []
            max_display_lines = 15
            
            self._clear_screen()
            print("=" * 80)
            print("📺 whisper.cpp 流式转写")
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
                    
                    # whisper.cpp 流式转写
                    # 注意：pywhispercpp 的 transcribe 方法接受音频数组
                    result = self.model.transcribe(audio_chunk)
                    
                    elapsed = time.perf_counter() - start_time
                    
                    if result:
                        text = result.strip()
                        if text:
                            timestamp = time.strftime("%H:%M:%S")
                            all_sentences.append((timestamp, text))
                            
                            self._refresh_display(all_sentences, max_display_lines)
                            
                            logger.debug(f"处理性能: {elapsed*1000:.0f}ms, 音频能量: {audio_energy:.4f}")
                    
                    chunk_index += 1
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ 处理块 {chunk_index} 时出错: {e}")
                    continue
            
            # 处理结束
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
                print(f"总计：{len(all_sentences)} 段，{sum(len(t[1]) for t in all_sentences)} 字")
                print("=" * 80 + "\n")
        
        except Exception as e:
            logger.error(f"❌ 音频处理失败: {e}")
            self.is_processing = False
    
    def _clear_screen(self):
        """清空屏幕"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _refresh_display(self, all_sentences, max_lines):
        """刷新显示"""
        self._clear_screen()
        
        total_chars = sum(len(t[1]) for t in all_sentences)
        
        print("=" * 80)
        print("📺 whisper.cpp 实时转写")
        print("=" * 80)
        print(f"已识别：{len(all_sentences)} 段，{total_chars} 字")
        print("=" * 80)
        print()
        
        display_sentences = all_sentences[-max_lines:] if len(all_sentences) > max_lines else all_sentences
        
        if len(all_sentences) > max_lines:
            print(f"... (前面还有 {len(all_sentences) - max_lines} 段)")
            print()
        
        for timestamp, text in display_sentences:
            print(f"[{timestamp}] {text}")
        
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
        logger.info("🎤 whisper.cpp 流式转写已启动！")
        logger.info("=" * 80)
        logger.info("提示：")
        logger.info("  - 播放任何音频（音乐、视频、网页等），转写结果会实时显示")
        logger.info("  - whisper.cpp 使用 3-5 秒的音频块，延迟会比 Paraformer 高")
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
        logger.info("✅ whisper.cpp 流式转写已停止")
        logger.info("=" * 80)


def main():
    """主函数"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "whisper.cpp 流式转写测试" + " " * 20 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    print("模型选择：")
    print("  1. tiny   - 最快，准确率较低（~75 MB）")
    print("  2. base   - 平衡（~142 MB）【推荐】")
    print("  3. small  - 准确率较高（~466 MB）")
    print("  4. medium - 准确率高（~1.5 GB）")
    print("  5. large  - 最准确（~2.9 GB）")
    print()
    
    model_choice = input("请选择模型（直接回车使用 base）: ").strip()
    
    model_map = {
        "1": "tiny",
        "2": "base",
        "3": "small",
        "4": "medium",
        "5": "large",
        "": "base"
    }
    
    model_name = model_map.get(model_choice, "base")
    
    print(f"\n使用模型: {model_name}\n")
    
    streaming = RealtimeAudioWhisperCpp(
        sample_rate=16000,
        chunk_duration=3.0,  # whisper.cpp 建议 3-5 秒
        model_name=model_name,
        device="cpu"  # whisper.cpp 主要使用 CPU
    )
    
    streaming.start(duration=None)


if __name__ == "__main__":
    main()
