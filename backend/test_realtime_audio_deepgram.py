"""
实时音频捕获 + Deepgram Nova-2 流式转写测试

使用 Deepgram 的流式 API 进行实时语音识别
Nova-2 是 Deepgram 的最新模型，支持真正的流式转写

依赖安装：
pip install pyaudiowpatch deepgram-sdk

使用方法：
python test_realtime_audio_deepgram.py
"""

import os
import sys
import time
import queue
import logging
import threading
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RealtimeAudioDeepgram:
    """实时音频流式转写（Deepgram Nova-2）"""
    
    def __init__(
        self,
        sample_rate: int = 48000,  # 使用设备支持的采样率
        model: str = "nova-2",  # nova-2 支持中文
    ):
        """
        初始化实时音频流式转写
        
        Args:
            sample_rate: 采样率（Hz），使用 48000 匹配音频设备
            model: 模型名称（nova-2 支持中文）
        """
        self.sample_rate = sample_rate
        self.model = model
        
        # 音频队列
        self.audio_queue = queue.Queue()
        
        # 控制标志
        self.is_recording = False
        self.is_processing = False
        
        # PyAudio 对象
        self.pyaudio = None
        self.stream = None
        
        # Deepgram 连接
        self.dg_connection = None
        self.connection_context = None
        
        # 转写结果
        self.all_sentences = []
        self.current_sentence = ""
        
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
            from deepgram import DeepgramClient
            from deepgram.core.events import EventType
            from deepgram.listen.v1.types import ListenV1Results
            logger.info(f"✅ deepgram-sdk 已安装")
            self.DeepgramClient = DeepgramClient
            self.EventType = EventType
            self.ListenV1Results = ListenV1Results
        except ImportError as e:
            logger.error(f"导入失败: {e}")
            raise ImportError(
                "请安装 deepgram-sdk 库:\n"
                "pip install deepgram-sdk\n"
                f"错误详情: {e}"
            )
    
    def _setup_deepgram(self):
        """设置 Deepgram 流式连接"""
        logger.info("=" * 80)
        logger.info("正在连接 Deepgram 流式 API...")
        logger.info("=" * 80)
        logger.info(f"模型: {self.model}")
        logger.info(f"采样率: {self.sample_rate} Hz")
        
        try:
            # 创建 Deepgram 客户端（API key 通过环境变量 DEEPGRAM_API_KEY 传递）
            client = self.DeepgramClient()
            
            # 配置选项（v1 API，nova-2 支持中文）
            options = {
                "model": self.model,
                "language": "zh",  # 明确指定中文
                "encoding": "linear16",
                "sample_rate": str(self.sample_rate),
                "channels": 1,  # 单声道
            }
            
            # 使用 v1 API 创建连接
            self.dg_connection = client.listen.v1.connect(**options)
            self.connection_context = self.dg_connection.__enter__()
            
            # 设置事件处理器
            self.connection_context.on(self.EventType.OPEN, self._on_open)
            self.connection_context.on(self.EventType.MESSAGE, self._on_message)
            self.connection_context.on(self.EventType.ERROR, self._on_error)
            self.connection_context.on(self.EventType.CLOSE, self._on_close)
            
            # 在后台线程启动监听（start_listening 会阻塞）
            listen_thread = threading.Thread(
                target=self.connection_context.start_listening,
                daemon=True
            )
            listen_thread.start()
            
            logger.info("✅ Deepgram 连接成功")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Deepgram 连接失败: {e}")
            raise
    
    def _on_open(self, *args, **kwargs):
        """连接打开回调"""
        logger.info("🔗 Deepgram 连接已打开")
    
    def _on_message(self, message):
        """接收转写结果回调"""
        try:
            # 添加调试日志
            logger.info(f"收到消息类型: {type(message).__name__}")
            
            # v1 API 使用 ListenV1Results 类型
            if isinstance(message, self.ListenV1Results):
                logger.info(f"收到 ListenV1Results 消息")
                if message.channel and message.channel.alternatives:
                    transcript = message.channel.alternatives[0].transcript
                    is_final = message.is_final
                    
                    logger.info(f"转写文本: {transcript}, is_final: {is_final}")
                    
                    if not transcript:
                        return
                    
                    # 获取当前时间
                    current_time = time.strftime("%H:%M:%S")
                    
                    if is_final:
                        # 最终结果
                        self.all_sentences.append(f"[{current_time}] {transcript}")
                        self.current_sentence = ""
                        self._display_results()
                    else:
                        # 临时结果（实时显示）
                        self.current_sentence = f"[{current_time}] {transcript}"
                        self._display_results()
            else:
                logger.info(f"忽略消息类型: {type(message).__name__}")
                    
        except Exception as e:
            logger.error(f"处理转写结果时出错: {e}", exc_info=True)
    
    def _on_error(self, error):
        """错误回调"""
        logger.error(f"❌ Deepgram 错误: {error}")
    
    def _on_close(self, *args, **kwargs):
        """连接关闭回调"""
        logger.info("🔌 Deepgram 连接已关闭")
    
    def _display_results(self):
        """显示转写结果（覆盖式）"""
        self._clear_screen()
        
        print("=" * 80)
        print("📺 Deepgram Nova-2 实时转写")
        print("=" * 80)
        print()
        
        # 显示所有已完成的句子
        for sentence in self.all_sentences[-20:]:  # 只显示最近 20 句
            print(sentence)
        
        # 显示当前正在识别的句子（临时结果）
        if self.current_sentence:
            print()
            print("🎤 正在识别...")
            print(self.current_sentence)
        
        print()
        print("=" * 80)
        print(f"已识别：{len(self.all_sentences)} 句")
        print("按 Ctrl+C 停止")
        print("=" * 80)
    
    def _clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _get_loopback_device(self):
        """获取系统音频回环设备（WASAPI Loopback）"""
        p = self.pyaudio.PyAudio()
        
        try:
            # 获取默认 WASAPI 信息
            wasapi_info = p.get_host_api_info_by_type(self.pyaudio.paWASAPI)
            default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            
            # 检查是否支持 loopback
            if not default_speakers["isLoopbackDevice"]:
                # 尝试查找对应的 loopback 设备
                for loopback in p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        break
            
            logger.info(f"使用音频设备: {default_speakers['name']}")
            return default_speakers
            
        except Exception as e:
            logger.error(f"获取音频设备失败: {e}")
            raise
        finally:
            p.terminate()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频流回调函数"""
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (in_data, self.pyaudio.paContinue)
    
    def _capture_audio(self):
        """捕获系统音频"""
        device_info = self._get_loopback_device()
        
        p = self.pyaudio.PyAudio()
        
        try:
            self.stream = p.open(
                format=self.pyaudio.paInt16,
                channels=1,  # 单声道
                rate=self.sample_rate,
                input=True,
                input_device_index=device_info["index"],
                frames_per_buffer=1024,
                stream_callback=self._audio_callback,
            )
            
            logger.info("🎤 开始捕获系统音频...")
            self.stream.start_stream()
            
            # 保持流打开
            while self.is_recording:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"音频捕获失败: {e}")
            raise
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            p.terminate()
            logger.info("音频流已关闭")
    
    def _send_audio_to_deepgram(self):
        """将音频数据发送到 Deepgram"""
        logger.info("开始发送音频到 Deepgram...")
        
        try:
            while self.is_processing:
                try:
                    # 从队列获取音频数据（超时 0.1 秒）
                    audio_data = self.audio_queue.get(timeout=0.1)
                    
                    # 发送到 Deepgram
                    if self.connection_context:
                        self.connection_context.send_media(audio_data)
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"发送音频数据失败: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"音频发送线程异常: {e}")
        finally:
            logger.info("音频发送线程已停止")
    
    def start(self, duration: float = None):
        """
        开始实时转写
        
        Args:
            duration: 持续时间（秒），None 表示一直运行直到手动停止
        """
        self.is_recording = True
        self.is_processing = True
        
        # 设置 Deepgram 连接
        self._setup_deepgram()
        
        # 启动音频捕获线程
        capture_thread = threading.Thread(target=self._capture_audio, daemon=True)
        capture_thread.start()
        
        # 启动音频发送线程
        send_thread = threading.Thread(target=self._send_audio_to_deepgram, daemon=True)
        send_thread.start()
        
        logger.info("\n" + "=" * 80)
        logger.info("🎤 Deepgram Nova-2 实时转写已启动！")
        logger.info("=" * 80)
        logger.info("提示：")
        logger.info("  - 使用 Deepgram 云端 API")
        logger.info("  - 真正的流式转写，延迟极低")
        logger.info("  - 按 Ctrl+C 停止")
        logger.info("=" * 80)
        
        try:
            if duration:
                time.sleep(duration)
            else:
                # 一直运行直到 Ctrl+C
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n收到停止信号...")
        finally:
            self.stop()
    
    def stop(self):
        """停止转写"""
        logger.info("正在停止转写...")
        
        self.is_recording = False
        self.is_processing = False
        
        # 关闭 Deepgram 连接
        if self.dg_connection and self.connection_context:
            try:
                self.dg_connection.__exit__(None, None, None)
            except Exception as e:
                logger.error(f"关闭 Deepgram 连接失败: {e}")
        
        # 显示最终结果
        self._clear_screen()
        print("=" * 80)
        print("📺 Deepgram Nova-2 实时转写")
        print("=" * 80)
        print(f"已识别：{len(self.all_sentences)} 句")
        print("=" * 80)
        print()
        
        for sentence in self.all_sentences:
            print(sentence)
        
        print()
        print("=" * 80)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Deepgram 实时转写已停止")
        logger.info("=" * 80)


def main():
    """主函数"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 18 + "Deepgram Nova-2 实时转写测试" + " " * 18 + "║")
    print("║" + " " * 25 + "（云端 API）" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    streaming = RealtimeAudioDeepgram(
        sample_rate=48000,  # 使用设备支持的采样率
        model="nova-2",  # nova-2 支持中文
    )
    
    streaming.start(duration=None)


if __name__ == "__main__":
    main()
