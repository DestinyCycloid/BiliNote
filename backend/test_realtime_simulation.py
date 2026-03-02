"""
模拟实时直播场景 - 测试流式转写效果

模拟方式：
1. 读取本地音频文件
2. 按实时速度（1秒/块）逐块喂给模型
3. 实时打印识别结果，模拟直播字幕效果
"""

import os
import sys
import time
import soundfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from funasr import AutoModel


def simulate_realtime_transcription():
    """模拟实时直播转写"""
    
    print("=" * 80)
    print("模拟实时直播场景 - 流式转写测试")
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
        print("未找到测试音频文件")
        return
    
    print(f"使用音频: {test_audio}")
    print()
    
    # 加载模型
    print("加载 Paraformer-streaming 模型...")
    model = AutoModel(
        model="./models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        device="cpu",
        disable_update=True
    )
    print("模型加载完成")
    print()
    
    # 读取音频
    speech, sr = soundfile.read(test_audio)
    
    # 如果是立体声，转为单声道
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)
    
    # 重采样到 16kHz
    if sr != 16000:
        import numpy as np
        target_length = int(len(speech) * 16000 / sr)
        speech = np.interp(
            np.linspace(0, len(speech) - 1, target_length),
            np.arange(len(speech)),
            speech
        ).astype(np.float32)
        sr = 16000
    
    audio_duration = len(speech) / sr
    
    print(f"音频时长: {audio_duration:.2f}秒")
    print(f"采样率: {sr}Hz")
    print()
    
    # 流式配置
    chunk_size = [0, 17, 8]  # 1秒/块
    chunk_duration = 1.0  # 1秒
    chunk_samples = int(sr * chunk_duration)
    total_chunks = int(len(speech) / chunk_samples) + 1
    
    print(f"流式配置: {chunk_duration}秒/块，共 {total_chunks} 个块")
    print()
    print("=" * 80)
    print("开始模拟实时转写（按实时速度处理）")
    print("=" * 80)
    print()
    
    # 流式处理
    cache = {}
    full_text = ""
    total_processing_time = 0
    total_wait_time = 0
    
    simulation_start = time.time()
    
    for i in range(0, len(speech), chunk_samples):
        chunk_start_time = time.time()
        chunk_index = i // chunk_samples
        
        # 获取音频块
        chunk = speech[i:i+chunk_samples]
        is_final = (i + chunk_samples >= len(speech))
        
        # 转写
        process_start = time.time()
        res = model.generate(
            input=chunk,
            cache=cache,
            is_final=is_final,
            chunk_size=chunk_size,
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1
        )
        process_time = time.time() - process_start
        total_processing_time += process_time
        
        # 获取结果
        text = ""
        if res and len(res) > 0:
            text = res[0].get("text", "").strip()
            if text:
                full_text += text
        
        # 计算时间戳
        timestamp = i / sr
        
        # 实时打印（模拟直播字幕）
        status = "处理中" if not is_final else "完成"
        print(f"[{timestamp:6.1f}s] [{status}] "
              f"处理耗时: {process_time:.3f}s | "
              f"识别: {text if text else '(静音)'}")
        
        # 模拟实时：等待到这1秒过去
        elapsed = time.time() - chunk_start_time
        sleep_time = max(0, chunk_duration - elapsed)
        
        if sleep_time > 0 and not is_final:
            total_wait_time += sleep_time
            time.sleep(sleep_time)
    
    simulation_end = time.time()
    total_time = simulation_end - simulation_start
    
    print()
    print("=" * 80)
    print("模拟完成")
    print("=" * 80)
    print()
    print(f"音频时长: {audio_duration:.2f}秒")
    print(f"模拟总耗时: {total_time:.2f}秒")
    print(f"实际处理时间: {total_processing_time:.2f}秒")
    print(f"等待时间: {total_wait_time:.2f}秒")
    print(f"平均处理延迟: {total_processing_time / total_chunks:.3f}秒/块")
    print()
    print(f"实时性能:")
    print(f"  - 处理速度: {audio_duration / total_processing_time:.2f}x 实时")
    print(f"  - 模拟速度: {audio_duration / total_time:.2f}x 实时")
    print()
    print("完整识别文本:")
    print("-" * 80)
    print(full_text)
    print("-" * 80)
    print()
    
    # 评估实时性能
    avg_latency = total_processing_time / total_chunks
    if avg_latency < chunk_duration:
        print(f"结论: 可以实时处理！")
        print(f"  每块处理时间 ({avg_latency:.3f}s) < 块时长 ({chunk_duration}s)")
        print(f"  延迟约 {avg_latency * 1000:.0f}ms，适合直播场景")
    else:
        print(f"警告: 无法实时处理！")
        print(f"  每块处理时间 ({avg_latency:.3f}s) > 块时长 ({chunk_duration}s)")
        print(f"  会出现累积延迟")


if __name__ == "__main__":
    simulate_realtime_transcription()
