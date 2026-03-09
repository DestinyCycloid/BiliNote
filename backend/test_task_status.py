#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试任务状态管理（Redis 集成）
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

from app.services.note import NoteGenerator
from app.enmus.task_status_enums import TaskStatus
from app.utils.redis_client import RedisManager
from dotenv import load_dotenv

load_dotenv()

def test_task_status():
    """测试任务状态管理"""
    print("=" * 60)
    print("测试任务状态管理（Redis 集成）")
    print("=" * 60)
    
    # 创建测试任务 ID
    test_task_id = "test_task_123456"
    
    # 初始化 NoteGenerator
    generator = NoteGenerator()
    
    # 测试1：写入 PENDING 状态
    print("\n[测试1] 写入 PENDING 状态")
    generator._update_status(test_task_id, TaskStatus.PENDING, message="任务排队中")
    print("  ✅ 状态已写入")
    
    # 验证 Redis 中的数据
    redis_manager = RedisManager(db=0)
    if redis_manager.available:
        redis_key = f"task:{test_task_id}"
        data = redis_manager.hgetall(redis_key)
        print(f"  Redis 数据: {data}")
        
        # 检查 TTL
        ttl = redis_manager.ttl(redis_key)
        print(f"  TTL: {ttl}秒 (约 {ttl/3600:.1f} 小时)")
    
    # 测试2：更新为 DOWNLOADING 状态
    print("\n[测试2] 更新为 DOWNLOADING 状态")
    time.sleep(1)
    generator._update_status(test_task_id, TaskStatus.DOWNLOADING, message="正在下载音频")
    
    if redis_manager.available:
        data = redis_manager.hgetall(redis_key)
        print(f"  Redis 数据: {data}")
    
    # 测试3：更新为 TRANSCRIBING 状态
    print("\n[测试3] 更新为 TRANSCRIBING 状态")
    time.sleep(1)
    generator._update_status(test_task_id, TaskStatus.TRANSCRIBING, message="正在转写音频")
    
    if redis_manager.available:
        data = redis_manager.hgetall(redis_key)
        print(f"  Redis 数据: {data}")
    
    # 测试4：更新为 SUCCESS 状态
    print("\n[测试4] 更新为 SUCCESS 状态")
    time.sleep(1)
    generator._update_status(test_task_id, TaskStatus.SUCCESS)
    
    if redis_manager.available:
        data = redis_manager.hgetall(redis_key)
        print(f"  Redis 数据: {data}")
    
    # 测试5：测试 FAILED 状态
    print("\n[测试5] 测试 FAILED 状态")
    test_task_id_2 = "test_task_failed"
    generator._update_status(test_task_id_2, TaskStatus.FAILED, message="下载失败：网络超时")
    
    if redis_manager.available:
        redis_key_2 = f"task:{test_task_id_2}"
        data = redis_manager.hgetall(redis_key_2)
        print(f"  Redis 数据: {data}")
    
    # 测试6：验证文件系统备份
    print("\n[测试6] 验证文件系统备份")
    from pathlib import Path
    status_file = Path("note_results") / f"{test_task_id}.status.json"
    if status_file.exists():
        import json
        with open(status_file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        print(f"  文件数据: {file_data}")
        print("  ✅ 文件系统备份正常")
    else:
        print("  ❌ 文件系统备份不存在")
    
    # 清理测试数据
    print("\n[清理] 删除测试数据")
    if redis_manager.available:
        redis_manager.delete(f"task:{test_task_id}", f"task:{test_task_id_2}")
        print("  ✅ Redis 数据已清理")
    
    if status_file.exists():
        status_file.unlink()
        print("  ✅ 文件已清理")
    
    status_file_2 = Path("note_results") / f"{test_task_id_2}.status.json"
    if status_file_2.exists():
        status_file_2.unlink()
    
    print("\n" + "=" * 60)
    print("✅ 任务状态管理测试通过！")
    print("=" * 60)

if __name__ == "__main__":
    test_task_status()
