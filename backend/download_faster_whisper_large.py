"""
Faster-Whisper Large 模型下载脚本（ModelScope）

从 ModelScope 下载 Faster-Whisper large-v3 模型
"""

import os
from modelscope import snapshot_download


def download_faster_whisper_large():
    """下载 Faster-Whisper large-v3 模型（只下载 CTranslate2 格式）"""
    
    # 模型保存目录
    model_dir = "./models/whisper/whisper-large-v3"
    
    print("=" * 80)
    print("开始下载 Faster-Whisper large-v3 模型")
    print("=" * 80)
    print(f"保存路径: {model_dir}")
    print()
    
    # ModelScope 上的 Faster-Whisper large-v3 模型（CTranslate2 格式）
    model_id = "pengzhendong/faster-whisper-large-v3"
    
    print(f"模型 ID: {model_id}")
    print(f"模型大小: ~3 GB（仅 CTranslate2 格式）")
    print(f"参数量: 1550M")
    print()
    print("⚠️ 注意：只下载 Faster-Whisper 需要的文件，忽略其他格式")
    print()
    
    try:
        # 从 ModelScope 下载模型（CTranslate2 格式）
        snapshot_download(
            model_id=model_id,
            cache_dir="./models",
            local_dir=model_dir,
        )
        
        print()
        print("=" * 80)
        print("✅ Faster-Whisper large-v3 模型下载完成！")
        print("=" * 80)
        print()
        print("模型信息：")
        print("  - 参数量: 1550M")
        print("  - 实际下载: ~3 GB（仅 CTranslate2 格式）")
        print("  - 支持语言: 99 种语言")
        print("  - 准确率: 最高")
        print()
        print("使用方法：")
        print("  直接运行测试（脚本会自动使用 large-v3 模型）：")
        print("  python test_realtime_audio_faster_whisper.py")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print()
        print("备用方案：")
        print("1. 检查网络连接")
        print("2. 或手动从 ModelScope 下载：")
        print(f"   https://www.modelscope.cn/models/{model_id}")
        print("3. 只需要下载这些文件：")
        print("   - model.bin")
        print("   - config.json")
        print("   - tokenizer.json")
        print("   - vocabulary.txt")
        print()
        return False


def download_faster_whisper_medium():
    """下载 Faster-Whisper medium 模型（备选）"""
    
    model_dir = "./models/whisper/whisper-medium"
    
    print("=" * 80)
    print("开始下载 Faster-Whisper medium 模型")
    print("=" * 80)
    print(f"保存路径: {model_dir}")
    print()
    
    model_id = "pengzhendong/faster-whisper-medium"
    
    print(f"模型 ID: {model_id}")
    print(f"模型大小: ~1.5 GB")
    print(f"参数量: 769M")
    print()
    
    try:
        snapshot_download(
            model_id=model_id,
            cache_dir="./models",
            local_dir=model_dir,
        )
        
        print()
        print("=" * 80)
        print("✅ Faster-Whisper medium 模型下载完成！")
        print("=" * 80)
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def download_faster_whisper_small():
    """下载 Faster-Whisper small 模型（备选）"""
    
    model_dir = "./models/whisper/whisper-small"
    
    print("=" * 80)
    print("开始下载 Faster-Whisper small 模型")
    print("=" * 80)
    print(f"保存路径: {model_dir}")
    print()
    
    model_id = "pengzhendong/faster-whisper-small"
    
    print(f"模型 ID: {model_id}")
    print(f"模型大小: ~466 MB")
    print(f"参数量: 244M")
    print()
    
    try:
        snapshot_download(
            model_id=model_id,
            cache_dir="./models",
            local_dir=model_dir,
        )
        
        print()
        print("=" * 80)
        print("✅ Faster-Whisper small 模型下载完成！")
        print("=" * 80)
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def main():
    """主函数"""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 18 + "Faster-Whisper 模型下载" + " " * 18 + "║")
    print("║" + " " * 25 + "（ModelScope）" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    print("推荐下载 large-v3 模型（最大、最准确）")
    print()
    print("可用模型：")
    print("  1. small     - 准确率较高（~466 MB）")
    print("  2. medium    - 准确率高（~1.5 GB）")
    print("  3. large-v3  - 最准确（~3 GB）【推荐，默认】")
    print()
    
    choice = input("请选择要下载的模型（直接回车下载 large-v3）: ").strip()
    
    print()
    
    if choice == "1":
        success = download_faster_whisper_small()
        model_name = "small"
    elif choice == "2":
        success = download_faster_whisper_medium()
        model_name = "medium"
    else:
        # 默认下载 large-v3
        success = download_faster_whisper_large()
        model_name = "large-v3"
    
    if success:
        print()
        print("=" * 80)
        print("🎉 下载完成！")
        print("=" * 80)
        print()
        print("下一步：")
        print("  直接运行测试（脚本会自动使用 large-v3 模型）：")
        print("  python test_realtime_audio_faster_whisper.py")
        print()
    else:
        print()
        print("下载失败，请检查网络连接或手动下载")
        print()


if __name__ == "__main__":
    main()
