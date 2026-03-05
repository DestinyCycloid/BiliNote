# 卸载现有的 PyTorch CPU 版本...
pip uninstall -y torch torchvision torchaudio

# 安装 PyTorch CUDA 12.1 版本...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装支持流式的qwen-asr
pip install -U qwen-asr[vllm]