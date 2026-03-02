"""
测试 Paraformer-streaming 转写器 - 展示流式输出特性
"""

import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.logger import get_logger

logger = get_logger(__name__)


def test_paraformer_streaming():
    """测试流式输出 - 实时显示每个块的识别结果"""
    
    print("=" * 80)
    print("测试 Paraformer-streaming 流式转写")
    print("=" * 80)
    print()
    
    # 查找测试音频
    test_audio = None
    search_paths = [
        "./data/test_audio.mp3",
        "./uploads/*.wav",
        "./uploads/*.mp3",
    ]
    
    for path in search_paths:
        if "*" in path:
            import glob
            files = glob.glob(path)
            if files:
                test_audio = files[0]
                break
        elif os.path.exists(path):
            test_audio = path
            break
    
    if not test_audio:
        print("❌ 未找到测试音频文件")
        return
    
    print(f"使用测试音频: {test_audio}")
    print()
    
    # 手动实现流式处理，实时打印每个块的结果
    from funasr import AutoModel
    import soundfile
    
    print("加载 Paraformer-streaming 模型...")
    model = AutoModel(
        model="./models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        device="cpu",
        disable_update=True
    )
    print("模型加载完成")
    print()
    
    # 读取音频
    speech, sample_rate = soundfile.read(test_audio)
    audio_duration = len(speech) / sample_rate
    
    # 流式配置
    chunk_size = [0, 17, 8]  # 1020ms ≈ 1秒/块
    chunk_stride = chunk_size[1] * 960  # 17 * 960 = 16320 采样点
    total_chunk_num = int(len(speech) / chunk_stride + 1)
    
    print(f"音频时长: {audio_duration:.2f}秒")
    print(f"分为 {total_chunk_num} 个块进行流式处理")
    print(f"每个块约 {chunk_stride / sample_rate:.2f}秒")
    print()
    print("=" * 80)
    print("开始流式转写 - 实时输出每个块的识别结果")
    print("=" * 80)
    print()
    
    cache = {}
    full_text = ""
    chunk_count = 0
    start_time = time.perf_counter()
    
    for i in range(total_chunk_num):
        chunk_start_time = time.perf_counter()
        
        speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
        is_final = (i == total_chunk_num - 1)
        
        res = model.generate(
            input=speech_chunk,
            cache=cache,
            is_final=is_final,
            chunk_size=chunk_size,
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1
        )
        
        chunk_elapsed = time.perf_counter() - chunk_start_time
        
        if res and len(res) > 0:
            text = res[0].get("text", "").strip()
            if text:
                chunk_count += 1
                full_text += text
                
                # 实时打印每个块的结果
                time_pos = i * chunk_stride / sample_rate
                print(f"[块 {i+1:2d}/{total_chunk_num}] "
                      f"[{time_pos:5.2f}s] "
                      f"[耗时 {chunk_elapsed:.3f}s] "
                      f"→ {text}")
    
    total_elapsed = time.perf_counter() - start_time
    
    print()
    print("=" * 80)
    print("流式转写完成")
    print("=" * 80)
    print()
    print(f"总耗时: {total_elapsed:.2f}秒")
    print(f"音频时长: {audio_duration:.2f}秒")
    print(f"RTF: {total_elapsed / audio_duration:.3f}")
    print(f"有效块数: {chunk_count}/{total_chunk_num}")
    print()
    print("完整文本:")
    print("-" * 80)
    print(full_text)
    print("-" * 80)
    print()
    print("✅ 流式特性验证：")
    print("  - 每个块独立处理，实时输出结果")
    print("  - 不需要等待整个音频处理完成")
    print("  - 适合实时场景（如直播字幕）")
    print()


def test_paraformer():
    """测试 Paraformer-streaming 转写器（使用封装好的类）"""
    
    from app.transcriber.paraformer_streaming import ParaformerStreamingTranscriber
    
    print("=" * 80)
    print("测试 Paraformer-streaming 转写器")
    print("=" * 80)
    print()
    
    # 查找测试音频
    test_audio = None
    search_paths = [
        "./data/test_audio.mp3",
        "./uploads/*.wav",
        "./uploads/*.mp3",
    ]
    
    for path in search_paths:
        if "*" in path:
            import glob
            files = glob.glob(path)
            if files:
                test_audio = files[0]
                break
        elif os.path.exists(path):
            test_audio = path
            break
    
    if not test_audio:
        print("❌ 未找到测试音频文件")
        print("请将音频文件放到以下目录：")
        print("  - backend/data/")
        print("  - backend/uploads/")
        return
    
    print(f"使用测试音频: {test_audio}")
    print()
    
    # 创建转写器
    print("创建 Paraformer-streaming 转写器...")
    transcriber = ParaformerStreamingTranscriber(
        chunk_size=[0, 17, 8],  # 1秒/块（适合录制文件）
        use_vad=False,   # VAD 在流式模式下不兼容
        use_punc=False,  # 标点恢复在流式模式下不工作
        device="cpu"
    )
    print("✅ 转写器创建成功")
    print()
    
    # 执行转写
    print("开始转写...")
    print("-" * 80)
    
    try:
        result = transcriber.transcript(test_audio)
        
        print("-" * 80)
        print()
        print("=" * 80)
        print("转写结果")
        print("=" * 80)
        print()
        print(f"语言: {result.language}")
        print(f"文本长度: {len(result.full_text)} 字符")
        print(f"片段数: {len(result.segments)}")
        print()
        print("完整文本:")
        print("-" * 80)
        print(result.full_text)
        print("-" * 80)
        print()
        
        if result.segments:
            print("分段详情（前 5 个）:")
            print("-" * 80)
            for i, seg in enumerate(result.segments[:5]):
                print(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}")
            if len(result.segments) > 5:
                print(f"... 还有 {len(result.segments) - 5} 个分段")
            print("-" * 80)
        
        print()
        print("=" * 80)
        print("✅ 测试完成！")
        print("=" * 80)
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ 转写失败: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--streaming":
        # 流式演示模式
        test_paraformer_streaming()
    else:
        # 默认：完整测试
        print("提示：使用 --streaming 参数可以查看实时流式输出演示")
        print()
        test_paraformer()
