"""
Paraformer-streaming 模型下载脚本
"""

import os
from modelscope import snapshot_download

def download_paraformer_streaming():
    """下载 Paraformer-streaming 模型"""
    
    # 模型保存目录
    model_dir = "./models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
    
    print("=" * 80)
    print("开始下载 Paraformer-streaming 模型")
    print("=" * 80)
    print(f"保存路径: {model_dir}")
    print()
    
    # 从 ModelScope 下载模型
    try:
        snapshot_download(
            model_id="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
            cache_dir="./models",
            local_dir=model_dir,
        )
        print()
        print("=" * 80)
        print("✅ Paraformer-streaming 模型下载完成！")
        print("=" * 80)
        print()
        print("模型信息：")
        print("  - 参数量: 220M")
        print("  - 模型大小: ~880 MB")
        print("  - 支持语言: 中文（普通话）")
        print("  - 流式支持: ✅ 是")
        print("  - 延迟: 600ms（可配置）")
        print()
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        raise

def download_vad_model():
    """下载 VAD 模型（可选）"""
    
    model_dir = "./models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    
    print("=" * 80)
    print("下载 VAD 模型（语音活动检测）")
    print("=" * 80)
    print(f"保存路径: {model_dir}")
    print()
    
    try:
        snapshot_download(
            model_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            cache_dir="./models",
            local_dir=model_dir,
        )
        print()
        print("=" * 80)
        print("✅ VAD 模型下载完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("VAD 模型是可选的，可以继续使用")

def download_punc_model():
    """下载标点恢复模型（可选）"""
    
    model_dir = "./models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    
    print()
    print("=" * 80)
    print("下载标点恢复模型")
    print("=" * 80)
    print(f"保存路径: {model_dir}")
    print()
    
    try:
        snapshot_download(
            model_id="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            cache_dir="./models",
            local_dir=model_dir,
        )
        print()
        print("=" * 80)
        print("✅ 标点恢复模型下载完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("标点恢复模型是可选的，可以继续使用")

if __name__ == "__main__":
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "Paraformer-streaming 模型下载" + " " * 20 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # 下载主模型
    download_paraformer_streaming()
    
    # 询问是否下载可选模型
    print()
    download_optional = input("是否下载可选模型（VAD + 标点恢复）？[y/N]: ").strip().lower()
    
    if download_optional == 'y':
        download_vad_model()
        download_punc_model()
    
    print()
    print("=" * 80)
    print("🎉 所有模型下载完成！")
    print("=" * 80)
    print()
    print("现在可以使用 Paraformer-streaming 进行流式语音识别了。")
    print()
    print("使用方法：")
    print("  1. 修改 .env 文件：TRANSCRIBER_TYPE=paraformer-streaming")
    print("  2. 重启后端服务：npm run dev")
    print()
