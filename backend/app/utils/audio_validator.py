"""
音频文件完整性校验工具
"""
import os
from typing import Optional, Tuple


def validate_audio_file(file_path: str, min_size_kb: int = 10) -> Tuple[bool, str]:
    """
    快速校验音频文件完整性
    
    Args:
        file_path: 音频文件路径
        min_size_kb: 最小文件大小（KB），默认 10KB
        
    Returns:
        (是否有效, 错误信息)
    """
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        return False, "文件不存在"
    
    # 2. 检查文件大小（极快，< 1ms）
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "文件为空"
        
        if file_size < min_size_kb * 1024:
            return False, f"文件过小 ({file_size} bytes)"
    except Exception as e:
        return False, f"无法读取文件大小: {e}"
    
    # 3. 检查音频时长（快速，使用 mutagen）
    try:
        from mutagen import File
        audio = File(file_path)
        
        if audio is None:
            return False, "无法解析音频文件"
        
        # 获取时长
        duration = audio.info.length if hasattr(audio.info, 'length') else 0
        
        if duration < 0.5:  # 至少 0.5 秒
            return False, f"音频时长异常 ({duration:.2f}s)"
        
        return True, ""
        
    except ImportError:
        # 如果没有 mutagen，降级为只检查文件大小
        return True, ""
    except Exception as e:
        # 如果解析失败，可能是文件损坏
        return False, f"音频文件损坏: {e}"


def get_audio_duration(file_path: str) -> Optional[float]:
    """
    获取音频时长（秒）
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        时长（秒），失败返回 None
    """
    try:
        from mutagen import File
        audio = File(file_path)
        if audio and hasattr(audio.info, 'length'):
            return audio.info.length
    except:
        pass
    
    return None
