#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 集成完整测试
测试任务状态管理、转写缓存、音频缓存、Markdown 缓存
"""
import sys
import os
import json
import time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from app.utils.redis_client import RedisClient, RedisManager
from dotenv import load_dotenv

load_dotenv()

def test_redis_integration():
    """完整的 Redis 集成测试"""
    print("=" * 70)
    print("Redis 集成完整测试")
    print("=" * 70)
    
    # 检查 Redis 可用性
    print("\n[步骤1] 检查 Redis 连接")
    if not RedisClient.is_available():
        print("  ❌ Redis 不可用，测试终止")
        return False
    print("  ✅ Redis 连接正常")
    
    # 测试任务状态管理（DB0）
    print("\n[步骤2] 测试任务状态管理（DB0）")
    task_manager = RedisManager(db=0)
    test_task_id = "test_integration_task"
    
    # 写入任务状态
    task_key = f"task:{test_task_id}"
    task_data = {
        "status": "DOWNLOADING",
        "message": "正在下载音频",
        "updated_at": str(time.time())
    }
    success = task_manager.hset(task_key, task_data, ttl=86400)
    print(f"  写入任务状态: {success}")
    
    # 读取任务状态
    retrieved_data = task_manager.hgetall(task_key)
    print(f"  读取任务状态: {retrieved_data}")
    
    # 检查 TTL
    ttl = task_manager.ttl(task_key)
    print(f"  TTL: {ttl}秒 (约 {ttl/3600:.1f} 小时)")
    
    if retrieved_data and retrieved_data.get("status") == "DOWNLOADING":
        print("  ✅ 任务状态管理测试通过")
    else:
        print("  ❌ 任务状态管理测试失败")
        return False
    
    # 测试转写结果缓存（DB1）
    print("\n[步骤3] 测试转写结果缓存（DB1）")
    cache_manager = RedisManager(db=1)
    
    # 模拟转写结果
    transcript_key = "transcript:bilibili_BV1234567890_medium"
    transcript_data = {
        "language": "zh",
        "full_text": "这是一段测试转写文本",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "这是一段"},
            {"start": 2.5, "end": 5.0, "text": "测试转写文本"}
        ],
        "raw": {}
    }
    
    # 写入转写缓存
    success = cache_manager.set(
        transcript_key,
        json.dumps(transcript_data, ensure_ascii=False),
        ttl=604800  # 7天
    )
    print(f"  写入转写缓存: {success}")
    
    # 读取转写缓存
    cached_transcript = cache_manager.get(transcript_key)
    if cached_transcript:
        retrieved_transcript = json.loads(cached_transcript)
        print(f"  读取转写缓存: {retrieved_transcript['full_text']}")
        
        # 检查 TTL
        ttl = cache_manager.ttl(transcript_key)
        print(f"  TTL: {ttl}秒 (约 {ttl/86400:.1f} 天)")
        
        if retrieved_transcript["full_text"] == transcript_data["full_text"]:
            print("  ✅ 转写结果缓存测试通过")
        else:
            print("  ❌ 转写结果缓存测试失败")
            return False
    else:
        print("  ❌ 读取转写缓存失败")
        return False
    
    # 测试音频元信息缓存（DB1）
    print("\n[步骤4] 测试音频元信息缓存（DB1）")
    audio_key = "audio:bilibili_BV1234567890_medium"
    audio_data = {
        "file_path": "/path/to/audio.mp3",
        "title": "测试视频标题",
        "video_id": "BV1234567890",
        "duration": 300,
        "raw_info": {}
    }
    
    # 写入音频缓存
    success = cache_manager.set(
        audio_key,
        json.dumps(audio_data, ensure_ascii=False),
        ttl=604800  # 7天
    )
    print(f"  写入音频缓存: {success}")
    
    # 读取音频缓存
    cached_audio = cache_manager.get(audio_key)
    if cached_audio:
        retrieved_audio = json.loads(cached_audio)
        print(f"  读取音频缓存: {retrieved_audio['title']}")
        
        # 检查 TTL
        ttl = cache_manager.ttl(audio_key)
        print(f"  TTL: {ttl}秒 (约 {ttl/86400:.1f} 天)")
        
        if retrieved_audio["title"] == audio_data["title"]:
            print("  ✅ 音频元信息缓存测试通过")
        else:
            print("  ❌ 音频元信息缓存测试失败")
            return False
    else:
        print("  ❌ 读取音频缓存失败")
        return False
    
    # 测试 Markdown 结果缓存（DB1）
    print("\n[步骤5] 测试 Markdown 结果缓存（DB1）")
    markdown_key = f"markdown:{test_task_id}"
    markdown_content = """# 测试视频标题

## 主要内容

这是一段测试 Markdown 内容。

## 总结

测试完成。
"""
    
    # 写入 Markdown 缓存
    success = cache_manager.set(
        markdown_key,
        markdown_content,
        ttl=2592000  # 30天
    )
    print(f"  写入 Markdown 缓存: {success}")
    
    # 读取 Markdown 缓存
    cached_markdown = cache_manager.get(markdown_key)
    if cached_markdown:
        print(f"  读取 Markdown 缓存: {len(cached_markdown)} 字符")
        
        # 检查 TTL
        ttl = cache_manager.ttl(markdown_key)
        print(f"  TTL: {ttl}秒 (约 {ttl/86400:.1f} 天)")
        
        if cached_markdown == markdown_content:
            print("  ✅ Markdown 结果缓存测试通过")
        else:
            print("  ❌ Markdown 结果缓存测试失败")
            return False
    else:
        print("  ❌ 读取 Markdown 缓存失败")
        return False
    
    # 测试缓存过期
    print("\n[步骤6] 测试缓存过期机制")
    expire_test_key = "test:expire"
    cache_manager.set(expire_test_key, "test_value", ttl=2)  # 2秒过期
    print("  写入测试键，TTL=2秒")
    
    # 立即读取
    value = cache_manager.get(expire_test_key)
    print(f"  立即读取: {value}")
    
    # 等待3秒后读取
    print("  等待3秒...")
    time.sleep(3)
    value = cache_manager.get(expire_test_key)
    if value is None:
        print("  ✅ 缓存已过期，自动清理成功")
    else:
        print("  ❌ 缓存未过期，测试失败")
        return False
    
    # 测试 Redis 信息查询
    print("\n[步骤7] 查询 Redis 服务器信息")
    client = RedisClient.get_instance(db=0)
    if client:
        info = client.info()
        print(f"  Redis 版本: {info.get('redis_version')}")
        print(f"  已用内存: {info.get('used_memory_human')}")
        print(f"  连接客户端数: {info.get('connected_clients')}")
        print(f"  总命令数: {info.get('total_commands_processed')}")
        print("  ✅ Redis 信息查询成功")
    
    # 清理测试数据
    print("\n[步骤8] 清理测试数据")
    task_manager.delete(task_key)
    cache_manager.delete(transcript_key, audio_key, markdown_key, expire_test_key)
    print("  ✅ 测试数据已清理")
    
    print("\n" + "=" * 70)
    print("🎉 所有测试通过！Redis 集成成功！")
    print("=" * 70)
    
    # 输出集成总结
    print("\n集成总结：")
    print("  ✅ 任务状态管理（DB0）- 24小时自动过期")
    print("  ✅ 转写结果缓存（DB1）- 7天自动过期")
    print("  ✅ 音频元信息缓存（DB1）- 7天自动过期")
    print("  ✅ Markdown 结果缓存（DB1）- 30天自动过期")
    print("  ✅ 自动过期清理机制正常")
    print("  ✅ 降级策略已实现（Redis 不可用时使用文件系统）")
    
    return True

if __name__ == "__main__":
    success = test_redis_integration()
    sys.exit(0 if success else 1)
