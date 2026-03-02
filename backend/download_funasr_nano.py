"""
下载 Fun-ASR-Nano 模型到本地

使用方法:
    python download_funasr_nano.py
"""

from modelscope import snapshot_download

def download_funasr_nano():
    """下载 Fun-ASR-Nano-2512 模型"""
    print("开始下载 Fun-ASR-Nano-2512 模型...")
    print("模型大小约 1.6-3.2 GB，请耐心等待...")
    
    model_dir = snapshot_download(
        'FunAudioLLM/Fun-ASR-Nano-2512',
        cache_dir='./models',  # 下载到 backend/models 目录
        revision='master'
    )
    
    print(f"\n✅ 模型下载完成！")
    print(f"模型路径: {model_dir}")
    return model_dir

def download_vad_model():
    """下载 VAD 模型（语音活动检测）"""
    print("\n开始下载 VAD 模型 (fsmn-vad)...")
    
    vad_dir = snapshot_download(
        'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
        cache_dir='./models',
        revision='master'
    )
    
    print(f"✅ VAD 模型下载完成！")
    print(f"VAD 模型路径: {vad_dir}")
    return vad_dir

if __name__ == "__main__":
    try:
        # 下载主模型
        model_path = download_funasr_nano()
        
        # 下载 VAD 模型
        vad_path = download_vad_model()
        
        print("\n" + "="*60)
        print("🎉 所有模型下载完成！")
        print("="*60)
        print(f"\nFun-ASR-Nano 模型: {model_path}")
        print(f"VAD 模型: {vad_path}")
        print("\n现在可以启动后端服务进行测试了！")
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n请检查:")
        print("1. 是否已安装 modelscope: pip install modelscope")
        print("2. 网络连接是否正常")
        print("3. 磁盘空间是否充足（需要约 4-5 GB）")
