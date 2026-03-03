"""测试并行转写 vs 串行转写的性能对比"""
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.transcriber.funasr_nano import FunASRNanoTranscriber

# 测试音频文件（约 1 分钟）
TEST_AUDIO = "C:\\Home\\demo\\BiliNote\\backend\\data\\data\\BV124AqzgEWz_p1.mp3"

def test_single_transcribe(transcriber, audio_file, index):
    """单个转写任务"""
    start = time.time()
    result = transcriber.transcript(audio_file)
    duration = time.time() - start
    print(f"  [{index}] 完成，耗时: {duration:.2f}秒，文本长度: {len(result.full_text)}")
    return duration

def test_parallel(transcriber, audio_file, count=4):
    """并行转写测试"""
    print(f"\n{'='*60}")
    print(f"并行测试：{count} 个任务同时执行")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [
            executor.submit(test_single_transcribe, transcriber, audio_file, i+1)
            for i in range(count)
        ]
        durations = [f.result() for f in futures]
    
    total_time = time.time() - start_time
    
    print(f"\n并行结果:")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  平均每个: {sum(durations)/len(durations):.2f}秒")
    print(f"  最快: {min(durations):.2f}秒")
    print(f"  最慢: {max(durations):.2f}秒")
    
    return total_time, durations

def test_serial(transcriber, audio_file, count=4):
    """串行转写测试"""
    print(f"\n{'='*60}")
    print(f"串行测试：{count} 个任务依次执行")
    print(f"{'='*60}")
    
    start_time = time.time()
    durations = []
    
    for i in range(count):
        duration = test_single_transcribe(transcriber, audio_file, i+1)
        durations.append(duration)
    
    total_time = time.time() - start_time
    
    print(f"\n串行结果:")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  平均每个: {sum(durations)/len(durations):.2f}秒")
    
    return total_time, durations

def main():
    print("="*60)
    print("FunASR-Nano 并行 vs 串行性能测试")
    print("="*60)
    
    # 初始化转写器
    print("\n初始化转写器（使用 CUDA）...")
    transcriber = FunASRNanoTranscriber(device="cuda")
    
    # 预热：加载模型
    print("预热：加载模型...")
    transcriber._load_model()
    print("✅ 模型加载完成\n")
    
    # 测试参数
    test_count = 4
    
    # 串行测试
    serial_time, serial_durations = test_serial(transcriber, TEST_AUDIO, test_count)
    
    # 等待一下，让 GPU 冷却
    print("\n等待 3 秒...")
    time.sleep(3)
    
    # 并行测试
    parallel_time, parallel_durations = test_parallel(transcriber, TEST_AUDIO, test_count)
    
    # 对比结果
    print(f"\n{'='*60}")
    print("性能对比")
    print(f"{'='*60}")
    print(f"串行总耗时: {serial_time:.2f}秒")
    print(f"并行总耗时: {parallel_time:.2f}秒")
    print(f"加速比: {serial_time / parallel_time:.2f}x")
    print(f"时间节省: {serial_time - parallel_time:.2f}秒 ({(serial_time - parallel_time) / serial_time * 100:.1f}%)")
    
    # 分析
    print(f"\n分析:")
    avg_serial = sum(serial_durations) / len(serial_durations)
    avg_parallel = sum(parallel_durations) / len(parallel_durations)
    print(f"  串行平均单个耗时: {avg_serial:.2f}秒")
    print(f"  并行平均单个耗时: {avg_parallel:.2f}秒")
    print(f"  单个任务变慢: {avg_parallel / avg_serial:.2f}x ({(avg_parallel - avg_serial) / avg_serial * 100:.1f}%)")
    
    if parallel_time < serial_time:
        print(f"\n✅ 并行更快！建议使用并行处理")
    else:
        print(f"\n⚠️ 并行更慢！建议使用串行处理")

if __name__ == "__main__":
    main()
