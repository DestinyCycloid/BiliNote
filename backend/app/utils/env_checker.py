# 缓存 torch 模块，避免重复导入导致 cuDNN 问题
_torch_module = None

def _get_torch():
    """获取 torch 模块（单例模式）"""
    global _torch_module
    if _torch_module is None:
        try:
            import torch
            _torch_module = torch
        except ImportError:
            pass
    return _torch_module

def is_cuda_available() -> bool:
    torch = _get_torch()
    if torch is None:
        return False
    return torch.cuda.is_available()

def is_torch_installed() -> bool:
    return _get_torch() is not None

