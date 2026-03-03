"""
实时音频捕获 + Paraformer-streaming 流式转写测试

功能：
1. 实时捕获 Windows 系统音频（浏览器播放的视频、音乐等）
2. 使用 Paraformer-streaming 模型进行流式转写
3. 实时显示字幕效果
4. 启用 CUDA 加速

依赖安装：
pip install pyaudiowpatch numpy funasr torch torchaudio

使用方法：
python test_realtime_audio_streaming.py

使用场景：
- 在浏览器播放视频，程序会实时显示字幕
- 播放音乐，程序会实时显示歌词
- 播放任何系统音频，都会实时转写

技术说明：
- 使用 PyAudioWPatch 库（PyAudio 的 Windows 补丁版本）
- 通过 WASAPI loopback 捕获系统音频
- 无需启用"立体声混音"等设置
"""

import sys
import os
import time
import threading
import queue
import numpy as np
from typing import Optional
from collections import deque
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.logger import get_logger

logger = get_logger(__name__)


class RealtimeAudioStreaming:
    """实时音频流式转写（Windows 系统音频捕获）"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 2,  # 每个块的时长（秒）
        device: str = "cuda"
    ):
        """
        初始化实时音频流式转写
        
        Args:
            sample_rate: 采样率（Hz），Paraformer 需要 16000Hz
            chunk_duration: 每个音频块的时长（秒），建议 0.6-2.0 秒
            device: 运行设备（cuda 或 cpu）
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)  # 采样点数
        self.device = device
        
        # 音频队列
        self.audio_queue = queue.Queue()
        
        # 控制标志
        self.is_recording = False
        self.is_processing = False
        
        # 模型
        self.model = None
        self.cache = {}
        
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
                "请安装 pyaudiowpatch 库（PyAudio 的 Windows 补丁版本）:\n"
                "pip install pyaudiowpatch\n\n"
                "这个库专为 Windows 设计，支持 WASAPI loopback 捕获系统音频"
            )
        
        try:
            import funasr
            logger.info(f"✅ funasr 已安装，版本: {funasr.__version__}")
        except ImportError:
            raise ImportError(
                "请安装 funasr 库:\n"
                "pip install funasr"
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
            raise ImportError(
                "请安装 PyTorch:\n"
                "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
            )
    
    def _load_model(self):
        """加载 Paraformer-streaming 模型（参考项目实现）"""
        if self.model is not None:
            return
        
        from funasr import AutoModel
        
        logger.info("=" * 80)
        logger.info("正在加载 Paraformer-streaming 模型...")
        logger.info("=" * 80)
        
        # 优先使用本地模型（与项目保持一致）
        local_model_dir = "./models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
        
        if os.path.exists(local_model_dir):
            model_path = local_model_dir
            logger.info(f"使用本地模型: {model_path}")
        else:
            model_path = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
            logger.info(f"本地模型不存在，从 ModelScope 下载: {model_path}")
            logger.info("提示：首次运行会自动下载模型，请耐心等待...")
        
        # 构建模型参数（与项目保持一致）
        model_kwargs = {
            "model": model_path,
            "device": self.device,
            "disable_update": True,
        }
        
        # 加载模型
        self.model = AutoModel(**model_kwargs)
        
        logger.info(f"✅ Paraformer-streaming 模型加载完成，设备: {self.device}")
        logger.info(f"块大小: {self.chunk_size} 采样点 = {self.chunk_duration:.2f} 秒")
        logger.info("=" * 80)
    
    def _list_audio_devices(self):
        """列出所有音频设备"""
        p = self.pyaudio.PyAudio()
        
        logger.info("\n" + "=" * 80)
        logger.info("可用的音频设备：")
        logger.info("=" * 80)
        
        default_speakers = None
        loopback_device = None
        
        # 查找所有设备
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            
            # 查找默认扬声器的 loopback 设备
            if info.get("isLoopbackDevice"):
                logger.info(f"\n🔊 找到 Loopback 设备（系统音频捕获）:")
                logger.info(f"  设备 ID: {i}")
                logger.info(f"  设备名称: {info['name']}")
                logger.info(f"  采样率: {int(info['defaultSampleRate'])}Hz")
                logger.info(f"  输入声道: {info['maxInputChannels']}")
                loopback_device = info
        
        if not loopback_device:
            logger.warning("\n⚠️ 未找到 Loopback 设备")
            logger.info("正在查找默认扬声器...")
            
            # 查找默认扬声器
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
        """
        音频捕获线程（使用 WASAPI loopback）
        """
        try:
            logger.info("=" * 80)
            logger.info("🔊 使用 WASAPI Loopback 模式捕获系统音频")
            logger.info("=" * 80)
            
            # 获取 loopback 设备
            p = self.pyaudio.PyAudio()
            
            try:
                # 获取默认扬声器的 loopback 设备
                wasapi_info = p.get_default_wasapi_loopback()
                logger.info(f"扬声器设备: {wasapi_info['name']}")
                
                # 获取设备的原始采样率
                device_sample_rate = int(wasapi_info['defaultSampleRate'])
                logger.info(f"设备采样率: {device_sample_rate}Hz")
                logger.info(f"目标采样率: {self.sample_rate}Hz")
                
                # 计算设备的块大小
                device_chunk_size = int(device_sample_rate * self.chunk_duration)
                
                logger.info(f"开始捕获系统音频")
                logger.info(f"每个块: {self.chunk_duration:.2f} 秒")
                logger.info(f"设备块大小: {device_chunk_size} 采样点")
                logger.info(f"目标块大小: {self.chunk_size} 采样点")
                logger.info("提示：播放任何音频（音乐、视频等），转写结果会实时显示")
                logger.info("=" * 80 + "\n")
                
                # 打开音频流
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
                        # 读取音频数据
                        audio_data = self.stream.read(device_chunk_size, exception_on_overflow=False)
                        
                        # 转换为 numpy 数组
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        
                        # 转换为 float32 并归一化到 [-1, 1]
                        audio_array = audio_array.astype(np.float32) / 32768.0
                        
                        # 如果是多声道，转换为单声道
                        if wasapi_info['maxInputChannels'] > 1:
                            audio_array = audio_array.reshape(-1, wasapi_info['maxInputChannels'])
                            audio_array = audio_array.mean(axis=1)
                        
                        # 重采样到目标采样率（如果需要）
                        if device_sample_rate != self.sample_rate:
                            # 简单的线性插值重采样
                            target_length = int(len(audio_array) * self.sample_rate / device_sample_rate)
                            audio_array = np.interp(
                                np.linspace(0, len(audio_array) - 1, target_length),
                                np.arange(len(audio_array)),
                                audio_array
                            ).astype(np.float32)
                        
                        # 放入队列
                        self.audio_queue.put(audio_array)
                        
                    except Exception as e:
                        logger.error(f"读取音频数据时出错: {e}")
                        continue
                
            finally:
                # 关闭音频流
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                p.terminate()
                logger.info("音频流已关闭")
        
        except Exception as e:
            logger.error(f"❌ 音频捕获失败: {e}")
            logger.error("提示：")
            logger.error("  1. 确保已安装 pyaudiowpatch: pip install pyaudiowpatch")
            logger.error("  2. 确保系统正在播放音频")
            logger.error("  3. 检查 Windows 音频设置")
            self.is_recording = False
    
    def _audio_processing_thread(self):
        """音频处理线程（流式转写）- 实时累积显示"""
        try:
            chunk_index = 0
            all_sentences = []  # 保存完整的句子
            current_line = []  # 当前行正在累积的片段
            current_line_start_time = None  # 当前行的开始时间
            max_display_lines = 15  # 屏幕最多显示多少行
            
            # 初始化显示
            self._clear_screen()
            print("=" * 80)
            print("📺 实时字幕（实时累积显示）")
            print("=" * 80)
            print()
            
            while self.is_processing:
                try:
                    # 从队列获取音频块（超时 1 秒）
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                    
                    # 检查音频能量（避免处理静音）
                    audio_energy = np.abs(audio_chunk).mean()
                    if audio_energy < 0.001:  # 静音阈值
                        chunk_index += 1
                        continue
                    
                    # 流式转写
                    start_time = time.perf_counter()
                    
                    res = self.model.generate(
                        input=audio_chunk,
                        cache=self.cache,
                        is_final=False,  # 实时模式，永远不是最后一个块
                        chunk_size=[0, 10, 5],  # 600ms 低延迟配置
                        encoder_chunk_look_back=4,
                        decoder_chunk_look_back=1,
                    )
                    
                    elapsed = time.perf_counter() - start_time
                    
                    # 处理识别结果
                    if res and len(res) > 0:
                        text = res[0].get("text", "").strip()
                        if text:
                            timestamp = time.strftime("%H:%M:%S")
                            
                            # 如果是新行的开始，记录时间
                            if not current_line:
                                current_line_start_time = timestamp
                            
                            # 累积到当前行
                            current_line.append(text)
                            current_text = "".join(current_line)
                            
                            # 判断是否需要换行（达到一定长度或遇到标点）
                            should_newline = (
                                len(current_text) >= 40 or  # 超过 40 字换行
                                any(p in text for p in ['。', '！', '？'])  # 遇到句号换行
                            )
                            
                            if should_newline:
                                # 保存完整的句子
                                all_sentences.append((current_line_start_time, current_text))
                                current_line = []
                                current_line_start_time = None
                            
                            # 刷新显示（包括当前正在累积的行）
                            self._refresh_display_with_current(
                                all_sentences, 
                                current_line, 
                                current_line_start_time,
                                max_display_lines
                            )
                            
                            # 性能信息（调试用）
                            if chunk_index % 10 == 0:
                                logger.debug(f"处理性能: {elapsed*1000:.0f}ms, 音频能量: {audio_energy:.4f}")
                    
                    chunk_index += 1
                    
                except queue.Empty:
                    # 队列为空，继续等待
                    continue
                except Exception as e:
                    logger.error(f"❌ 处理块 {chunk_index} 时出错: {e}")
                    continue
            
            # 处理结束，保存剩余的行
            if current_line:
                current_text = "".join(current_line)
                all_sentences.append((current_line_start_time, current_text))
            
            # 显示完整内容
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
        """刷新显示所有内容（包括当前正在累积的行）"""
        # 清空屏幕
        self._clear_screen()
        
        # 显示标题
        total_chars = sum(len(t[1]) for t in all_sentences)
        if current_line:
            total_chars += len("".join(current_line))
        
        print("=" * 80)
        print("📺 实时字幕")
        print("=" * 80)
        print(f"已识别：{len(all_sentences)} 句，{total_chars} 字")
        print("=" * 80)
        print()
        
        # 显示最近的完整句子
        display_sentences = all_sentences[-max_lines:] if len(all_sentences) > max_lines else all_sentences
        
        if len(all_sentences) > max_lines:
            print(f"... (前面还有 {len(all_sentences) - max_lines} 句)")
            print()
        
        for timestamp, text in display_sentences:
            print(f"[{timestamp}] {text}")
        
        # 显示当前正在累积的行（用不同颜色或标记）
        if current_line:
            current_text = "".join(current_line)
            print(f"[{current_time}] {current_text} ...")  # 加 ... 表示还在继续
        
        print()
        print("-" * 80)
        print("提示：按 Ctrl+C 停止录音")
        print("-" * 80)
    
    def start(self, duration: Optional[float] = None):
        """
        开始实时转写（仅捕获系统音频）
        
        Args:
            duration: 录音时长（秒），None 表示无限录音
        """
        # 列出设备
        loopback_device = self._list_audio_devices()
        
        if not loopback_device:
            logger.error("❌ 未找到可用的 Loopback 设备，无法捕获系统音频")
            logger.error("请确保：")
            logger.error("  1. 已安装 pyaudiowpatch: pip install pyaudiowpatch")
            logger.error("  2. Windows 音频驱动正常工作")
            return
        
        # 加载模型
        self._load_model()
        
        # 重置缓存
        self.cache = {}
        
        # 启动录音线程（强制使用 loopback 模式捕获系统音频）
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
        logger.info("🎤 实时转写已启动！（仅捕获系统音频）")
        logger.info("=" * 80)
        logger.info("提示：")
        logger.info("  - 播放任何音频（音乐、视频、网页等），转写结果会实时显示")
        logger.info("  - 不会使用麦克风，只捕获扬声器输出")
        logger.info("  - 按 Ctrl+C 停止录音")
        if duration:
            logger.info(f"  - 将在 {duration} 秒后自动停止")
        logger.info("=" * 80 + "\n")
        
        try:
            if duration:
                # 定时停止
                time.sleep(duration)
                logger.info(f"\n⏰ 录音时长已达 {duration} 秒，停止录音...")
            else:
                # 无限录音，等待 Ctrl+C
                while self.is_recording:
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.info("\n\n⏹️  收到停止信号，正在停止...")
        
        finally:
            # 停止录音和处理
            self.stop()
    
    def stop(self):
        """停止实时转写"""
        self.is_recording = False
        self.is_processing = False
        
        # 等待线程结束
        time.sleep(1)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ 实时转写已停止")
        logger.info("=" * 80)


def main():
    """主函数"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "实时音频流式转写测试" + " " * 20 + "║")
    print("║" + " " * 15 + "Paraformer-streaming + CUDA 加速" + " " * 15 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # 创建实时转写器
    streaming = RealtimeAudioStreaming(
        sample_rate=16000,
        chunk_duration=0.6,  # 600ms 低延迟
        device="cuda"  # 使用 CUDA 加速
    )
    
    # 开始实时转写
    # 参数说明：
    #   - duration: 录音时长（秒），None 表示无限录音
    # 注意：仅捕获系统音频（扬声器输出），不使用麦克风
    streaming.start(
        duration=None  # 无限录音，按 Ctrl+C 停止
    )


if __name__ == "__main__":
    main()
