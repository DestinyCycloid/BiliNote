import os
import platform
from enum import Enum
import threading

# 不再在模块级别导入所有 transcriber，改为按需导入
# from app.transcriber.groq import GroqTranscriber
# from app.transcriber.whisper import WhisperTranscriber
# from app.transcriber.bcut import BcutTranscriber
# from app.transcriber.kuaishou import KuaishouTranscriber
# from app.transcriber.deepgram import DeepgramTranscriber
# from app.transcriber.funasr_nano import FunASRNanoTranscriber
# from app.transcriber.paraformer_streaming import ParaformerStreamingTranscriber
from app.utils.logger import get_logger

logger = get_logger(__name__)

class TranscriberType(str, Enum):
    FAST_WHISPER = "fast-whisper"
    MLX_WHISPER = "mlx-whisper"
    BCUT = "bcut"
    KUAISHOU = "kuaishou"
    GROQ = "groq"
    DEEPGRAM = "deepgram"
    FUNASR_NANO = "funasr-nano"
    PARAFORMER_STREAMING = "paraformer-streaming"

# 仅在 Apple 平台启用 MLX Whisper
MLX_WHISPER_AVAILABLE = False
if platform.system() == "Darwin" and os.environ.get("TRANSCRIBER_TYPE") == "mlx-whisper":
    try:
        from app.transcriber.mlx_whisper_transcriber import MLXWhisperTranscriber
        MLX_WHISPER_AVAILABLE = True
        logger.info("MLX Whisper 可用，已导入")
    except ImportError:
        logger.warning("MLX Whisper 导入失败，可能未安装或平台不支持")

logger.info('初始化转录服务提供器')

# 转录器单例缓存
_transcribers = {
    TranscriberType.FAST_WHISPER: None,
    TranscriberType.MLX_WHISPER: None,
    TranscriberType.BCUT: None,
    TranscriberType.KUAISHOU: None,
    TranscriberType.GROQ: None,
    TranscriberType.DEEPGRAM: None,
    TranscriberType.FUNASR_NANO: None,
    TranscriberType.PARAFORMER_STREAMING: None,
}

# 线程锁，保护单例创建
_transcriber_lock = threading.Lock()

# 公共实例初始化函数
def _init_transcriber(key: TranscriberType, cls, *args, **kwargs):
    # 双重检查锁定模式
    if _transcribers[key] is None:
        with _transcriber_lock:
            # 再次检查，避免多个线程重复创建
            if _transcribers[key] is None:
                logger.info(f'创建 {cls.__name__} 实例: {key}')
                try:
                    _transcribers[key] = cls(*args, **kwargs)
                    logger.info(f'{cls.__name__} 创建成功')
                except Exception as e:
                    logger.error(f"{cls.__name__} 创建失败: {e}")
                    raise
    return _transcribers[key]

# 各类型获取方法
def get_groq_transcriber():
    from app.transcriber.groq import GroqTranscriber
    return _init_transcriber(TranscriberType.GROQ, GroqTranscriber)

def get_whisper_transcriber(model_size="base", device="cuda"):
    # 延迟导入，只在真正使用 whisper 时才导入
    from app.transcriber.whisper import WhisperTranscriber
    return _init_transcriber(TranscriberType.FAST_WHISPER, WhisperTranscriber, model_size=model_size, device=device)

def get_bcut_transcriber():
    from app.transcriber.bcut import BcutTranscriber
    return _init_transcriber(TranscriberType.BCUT, BcutTranscriber)

def get_kuaishou_transcriber():
    from app.transcriber.kuaishou import KuaishouTranscriber
    return _init_transcriber(TranscriberType.KUAISHOU, KuaishouTranscriber)

def get_mlx_whisper_transcriber(model_size="base"):
    if not MLX_WHISPER_AVAILABLE:
        logger.warning("MLX Whisper 不可用，请确保在 Apple 平台且已安装 mlx_whisper")
        raise ImportError("MLX Whisper 不可用")
    return _init_transcriber(TranscriberType.MLX_WHISPER, MLXWhisperTranscriber, model_size=model_size)

def get_deepgram_transcriber():
    """获取 Deepgram Whisper 转写器实例"""
    from app.transcriber.deepgram import DeepgramTranscriber
    return _init_transcriber(TranscriberType.DEEPGRAM, DeepgramTranscriber)

def get_funasr_nano_transcriber(model_size="2512", device="cuda"):
    """
    获取 Fun-ASR-Nano 转写器实例
    
    Args:
        model_size: 模型版本，默认 "2512" (2025年12月版本)
        device: 设备类型，"cuda" 或 "cpu"
    """
    from app.transcriber.funasr_nano import FunASRNanoTranscriber
    return _init_transcriber(
        TranscriberType.FUNASR_NANO,
        FunASRNanoTranscriber,
        model_size=model_size,
        device=device
    )

def get_paraformer_streaming_transcriber(use_vad=False, use_punc=False, device="cuda"):
    """
    获取 Paraformer-streaming 转写器实例
    
    Args:
        use_vad: 是否使用 VAD（语音活动检测）
        use_punc: 是否使用标点恢复
        device: 设备类型，"cuda" 或 "cpu"
    """
    from app.transcriber.paraformer_streaming import ParaformerStreamingTranscriber
    return _init_transcriber(
        TranscriberType.PARAFORMER_STREAMING,
        ParaformerStreamingTranscriber,
        use_vad=use_vad,
        use_punc=use_punc,
        device=device
    )

# 通用入口
def get_transcriber(transcriber_type="fast-whisper", model_size="base", device="cuda"):
    """
    获取指定类型的转录器实例

    参数:
        transcriber_type: 支持 "fast-whisper", "mlx-whisper", "bcut", "kuaishou", "groq"
        model_size: 模型大小，适用于 whisper 类
        device: 设备类型（如 cuda / cpu），仅 whisper 使用

    返回:
        对应类型的转录器实例
    """
    logger.info(f'请求转录器类型: {transcriber_type}')

    try:
        transcriber_enum = TranscriberType(transcriber_type)
    except ValueError:
        logger.warning(f'未知转录器类型 "{transcriber_type}"，默认使用 fast-whisper')
        transcriber_enum = TranscriberType.FAST_WHISPER

    whisper_model_size = os.environ.get("WHISPER_MODEL_SIZE", model_size)

    if transcriber_enum == TranscriberType.FAST_WHISPER:
        return get_whisper_transcriber(whisper_model_size, device=device)

    elif transcriber_enum == TranscriberType.MLX_WHISPER:
        if not MLX_WHISPER_AVAILABLE:
            logger.warning("MLX Whisper 不可用，回退到 fast-whisper")
            return get_whisper_transcriber(whisper_model_size, device=device)
        return get_mlx_whisper_transcriber(whisper_model_size)

    elif transcriber_enum == TranscriberType.BCUT:
        return get_bcut_transcriber()

    elif transcriber_enum == TranscriberType.KUAISHOU:
        return get_kuaishou_transcriber()

    elif transcriber_enum == TranscriberType.GROQ:
        return get_groq_transcriber()
    
    elif transcriber_enum == TranscriberType.DEEPGRAM:
        return get_deepgram_transcriber()
    
    elif transcriber_enum == TranscriberType.FUNASR_NANO:
        funasr_model_size = os.environ.get("FUNASR_NANO_MODEL_SIZE", "2512")
        return get_funasr_nano_transcriber(funasr_model_size, device=device)
    
    elif transcriber_enum == TranscriberType.PARAFORMER_STREAMING:
        use_vad = os.environ.get("PARAFORMER_USE_VAD", "false").lower() == "true"
        use_punc = os.environ.get("PARAFORMER_USE_PUNC", "false").lower() == "true"
        return get_paraformer_streaming_transcriber(use_vad=use_vad, use_punc=use_punc, device=device)

    # fallback
    logger.warning(f'未识别转录器类型 "{transcriber_type}"，使用 fast-whisper 作为默认')
    return get_whisper_transcriber(whisper_model_size, device=device)
