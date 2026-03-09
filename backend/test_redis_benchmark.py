#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 性能基准测试
对比 Redis 和文件系统的性能差异
"""
import sys
import os
import json
import time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from app.utils.redis_client import RedisManager
from dotenv import load_dotenv

load_dotenv()


def benchmark_redis_write(iterations=1000):
    """基准测试：Redis 写入性能"""
    manager = RedisManager(db=1)
    if not manager.available:
        return None
    
    start_time = time.time()
    
    for i in range(iterations):
        key = f"benchmark:write_{i}"
        data = {"index": i, "data": "test_value" * 10}
        manager.set(key, json.dumps(data), ttl=60)
    
    elapsed = time.time() - start_time
    
    # 清理
    keys = [f"benchmark:write_{i}" for i in range(iterations)]
    manager.delete(*keys)
    
    return elapsed


def benchmark_file_write(iterations=1000):
    """基准测试：文件系统写入性能"""
    test_dir = Path("benchmark_test")
    test_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    for i in range(iterations):
        file_path = test_dir / f"write_{i}.json"
        data = {"index": i, "data": "test_value" * 10}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    elapsed = time.time() - start_time
    
    # 清理
    for i in range(iterations):
        file_path = test_dir / f"write_{i}.json"
        if file_path.exists():
            file_path.unlink()
    test_dir.rmdir()
    
    return elapsed


def benchmark_redis_read(iterations=1000):
    """基准测试：Redis 读取性能"""
    manager = RedisManager(db=1)
    if not manager.available:
        return None
    
    # 准备数据
    for i in range(iterations):
        key = f"benchmark:read_{i}"
        data = {"index": i, "data": "test_value" * 10}
        manager.set(key, json.dumps(data), ttl=60)
    
    start_time = time.time()
    
    for i in range(iterations):
        key = f"benchmark:read_{i}"
        manager.get(key)
    
    elapsed = time.time() - start_time
    
    # 清理
    keys = [f"benchmark:read_{i}" for i in range(iterations)]
    manager.delete(*keys)
    
    return elapsed


def benchmark_file_read(iterations=1000):
    """基准测试：文件系统读取性能"""
    test_dir = Path("benchmark_test")
    test_dir.mkdir(exist_ok=True)
    
    # 准备数据
    for i in range(iterations):
        file_path = test_dir / f"read_{i}.json"
        data = {"index": i, "data": "test_value" * 10}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    start_time = time.time()
    
    for i in range(iterations):
        file_path = test_dir / f"read_{i}.json"
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
    
    elapsed = time.time() - start_time
    
    # 清理
    for i in range(iterations):
        file_path = test_dir / f"read_{i}.json"
        if file_path.exists():
            file_path.unlink()
    test_dir.rmdir()
    
    return elapsed


def benchmark_redis_hash(iterations=1000):
    """基准测试：Redis Hash 操作性能"""
    manager = RedisManager(db=0)
    if not manager.available:
        return None
    
    start_time = time.time()
    
    for i in range(iterations):
        key = f"benchmark:hash_{i}"
        data = {
            "status": "DOWNLOADING",
            "message": "正在下载",
            "updated_at": str(time.time())
        }
        manager.hset(key, data, ttl=60)
        manager.hgetall(key)
    
    elapsed = time.time() - start_time
    
    # 清理
    keys = [f"benchmark:hash_{i}" for i in range(iterations)]
    manager.delete(*keys)
    
    return elapsed


def benchmark_large_data():
    """基准测试：大数据处理"""
    print("\n[大数据测试] 1MB 数据读写")
    
    manager = RedisManager(db=1)
    large_data = "x" * 1000000  # 1MB
    
    # Redis 写入
    if manager.available:
        start = time.time()
        manager.set("benchmark:large", large_data, ttl=60)
        redis_write_time = time.time() - start
        
        # Redis 读取
        start = time.time()
        manager.get("benchmark:large")
        redis_read_time = time.time() - start
        
        manager.delete("benchmark:large")
    else:
        redis_write_time = None
        redis_read_time = None
    
    # 文件写入
    test_file = Path("benchmark_large.txt")
    start = time.time()
    test_file.write_text(large_data, encoding='utf-8')
    file_write_time = time.time() - start
    
    # 文件读取
    start = time.time()
    test_file.read_text(encoding='utf-8')
    file_read_time = time.time() - start
    
    test_file.unlink()
    
    print(f"  Redis 写入: {redis_write_time*1000:.2f}ms" if redis_write_time else "  Redis 不可用")
    print(f"  文件写入: {file_write_time*1000:.2f}ms")
    if redis_write_time:
        print(f"  写入提升: {file_write_time/redis_write_time:.1f}x")
    
    print(f"  Redis 读取: {redis_read_time*1000:.2f}ms" if redis_read_time else "  Redis 不可用")
    print(f"  文件读取: {file_read_time*1000:.2f}ms")
    if redis_read_time:
        print(f"  读取提升: {file_read_time/redis_read_time:.1f}x")


def main():
    """主测试函数"""
    print("=" * 70)
    print("Redis 性能基准测试")
    print("=" * 70)
    
    iterations = 1000
    print(f"\n测试迭代次数: {iterations}")
    
    # 测试1：写入性能
    print("\n[测试1] 写入性能对比")
    redis_write_time = benchmark_redis_write(iterations)
    file_write_time = benchmark_file_write(iterations)
    
    if redis_write_time:
        print(f"  Redis 写入: {redis_write_time:.3f}秒 ({iterations/redis_write_time:.0f} ops/s)")
        print(f"  文件写入: {file_write_time:.3f}秒 ({iterations/file_write_time:.0f} ops/s)")
        print(f"  性能提升: {file_write_time/redis_write_time:.1f}x")
        print(f"  平均延迟: Redis {redis_write_time*1000/iterations:.2f}ms vs 文件 {file_write_time*1000/iterations:.2f}ms")
    else:
        print("  ❌ Redis 不可用")
    
    # 测试2：读取性能
    print("\n[测试2] 读取性能对比")
    redis_read_time = benchmark_redis_read(iterations)
    file_read_time = benchmark_file_read(iterations)
    
    if redis_read_time:
        print(f"  Redis 读取: {redis_read_time:.3f}秒 ({iterations/redis_read_time:.0f} ops/s)")
        print(f"  文件读取: {file_read_time:.3f}秒 ({iterations/file_read_time:.0f} ops/s)")
        print(f"  性能提升: {file_read_time/redis_read_time:.1f}x")
        print(f"  平均延迟: Redis {redis_read_time*1000/iterations:.2f}ms vs 文件 {file_read_time*1000/iterations:.2f}ms")
    else:
        print("  ❌ Redis 不可用")
    
    # 测试3：Hash 操作性能
    print("\n[测试3] Hash 操作性能（任务状态管理场景）")
    redis_hash_time = benchmark_redis_hash(iterations)
    
    if redis_hash_time:
        print(f"  Redis Hash: {redis_hash_time:.3f}秒 ({iterations/redis_hash_time:.0f} ops/s)")
        print(f"  平均延迟: {redis_hash_time*1000/iterations:.2f}ms")
    else:
        print("  ❌ Redis 不可用")
    
    # 测试4：大数据处理
    benchmark_large_data()
    
    # 总结
    print("\n" + "=" * 70)
    print("性能测试总结")
    print("=" * 70)
    
    if redis_write_time and redis_read_time:
        print(f"\n✅ Redis 写入性能提升: {file_write_time/redis_write_time:.1f}x")
        print(f"✅ Redis 读取性能提升: {file_read_time/redis_read_time:.1f}x")
        print(f"✅ 平均响应时间: Redis {(redis_read_time+redis_write_time)*1000/(2*iterations):.2f}ms vs 文件 {(file_read_time+file_write_time)*1000/(2*iterations):.2f}ms")
        
        print("\n实际应用场景收益：")
        print(f"  - 任务状态查询（每秒100次）：节省 {(file_read_time-redis_read_time)*100:.1f}秒/秒")
        print(f"  - 缓存查询（每次）：节省 {(file_read_time-redis_read_time)*1000/iterations:.1f}ms")
        print(f"  - 状态更新（每次）：节省 {(file_write_time-redis_write_time)*1000/iterations:.1f}ms")
    else:
        print("\n❌ Redis 不可用，无法进行性能对比")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
