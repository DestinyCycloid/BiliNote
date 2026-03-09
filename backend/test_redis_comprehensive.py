#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 集成综合测试
覆盖所有场景：正常流程、边界情况、错误处理、并发测试、降级策略等
"""
import sys
import os
import json
import time
import threading
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from app.utils.redis_client import RedisClient, RedisManager
from app.services.note import NoteGenerator
from app.enmus.task_status_enums import TaskStatus
from dotenv import load_dotenv

load_dotenv()

class TestResults:
    """测试结果统计"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name):
        self.total += 1
        self.passed += 1
        print(f"  ✅ {test_name}")
    
    def add_fail(self, test_name, reason):
        self.total += 1
        self.failed += 1
        self.errors.append(f"{test_name}: {reason}")
        print(f"  ❌ {test_name}: {reason}")
    
    def summary(self):
        print("\n" + "=" * 70)
        print(f"测试总结: 总计 {self.total} 个测试")
        print(f"  ✅ 通过: {self.passed}")
        print(f"  ❌ 失败: {self.failed}")
        if self.errors:
            print("\n失败详情:")
            for error in self.errors:
                print(f"  - {error}")
        print("=" * 70)
        return self.failed == 0


def test_redis_connection(results):
    """测试1：Redis 连接测试"""
    print("\n[测试组1] Redis 连接测试")
    
    # 1.1 检查 Redis 可用性
    try:
        is_available = RedisClient.is_available()
        if is_available:
            results.add_pass("Redis 可用性检查")
        else:
            results.add_fail("Redis 可用性检查", "Redis 不可用")
            return False
    except Exception as e:
        results.add_fail("Redis 可用性检查", str(e))
        return False
    
    # 1.2 测试多 DB 连接
    try:
        db0 = RedisClient.get_instance(db=0)
        db1 = RedisClient.get_instance(db=1)
        if db0 and db1 and db0 != db1:
            results.add_pass("多 DB 连接")
        else:
            results.add_fail("多 DB 连接", "DB 实例创建失败")
    except Exception as e:
        results.add_fail("多 DB 连接", str(e))
    
    # 1.3 测试 PING
    try:
        if RedisClient.ping(db=0) and RedisClient.ping(db=1):
            results.add_pass("PING 测试")
        else:
            results.add_fail("PING 测试", "PING 失败")
    except Exception as e:
        results.add_fail("PING 测试", str(e))
    
    # 1.4 测试连接重置
    try:
        RedisClient.reset()
        is_available = RedisClient.is_available()
        if is_available:
            results.add_pass("连接重置")
        else:
            results.add_fail("连接重置", "重置后无法连接")
    except Exception as e:
        results.add_fail("连接重置", str(e))
    
    return True


def test_task_status_management(results):
    """测试2：任务状态管理"""
    print("\n[测试组2] 任务状态管理")
    
    generator = NoteGenerator()
    test_task_id = "test_comprehensive_task"
    redis_manager = RedisManager(db=0)
    
    # 2.1 写入各种状态
    statuses = [
        (TaskStatus.PENDING, "任务排队中"),
        (TaskStatus.PARSING, "解析链接"),
        (TaskStatus.DOWNLOADING, "下载中"),
        (TaskStatus.TRANSCRIBING, "转写中"),
        (TaskStatus.SUMMARIZING, "总结中"),
        (TaskStatus.SAVING, "保存中"),
        (TaskStatus.SUCCESS, None),
        (TaskStatus.FAILED, "测试失败信息"),
    ]
    
    for status, message in statuses:
        try:
            generator._update_status(test_task_id, status, message)
            time.sleep(0.1)
            
            # 验证 Redis 中的数据
            if redis_manager.available:
                redis_key = f"task:{test_task_id}"
                data = redis_manager.hgetall(redis_key)
                if data and data.get("status") == status.value:
                    results.add_pass(f"状态写入: {status.value}")
                else:
                    results.add_fail(f"状态写入: {status.value}", "状态不匹配")
            else:
                results.add_fail(f"状态写入: {status.value}", "Redis 不可用")
        except Exception as e:
            results.add_fail(f"状态写入: {status.value}", str(e))
    
    # 2.2 验证文件系统备份
    try:
        status_file = Path("note_results") / f"{test_task_id}.status.json"
        if status_file.exists():
            with open(status_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            if file_data.get("status") == TaskStatus.FAILED.value:
                results.add_pass("文件系统备份")
            else:
                results.add_fail("文件系统备份", "文件状态不匹配")
        else:
            results.add_fail("文件系统备份", "文件不存在")
    except Exception as e:
        results.add_fail("文件系统备份", str(e))
    
    # 2.3 测试 TTL
    try:
        if redis_manager.available:
            redis_key = f"task:{test_task_id}"
            ttl = redis_manager.ttl(redis_key)
            if 0 < ttl <= 86400:  # 24小时
                results.add_pass("TTL 设置正确")
            else:
                results.add_fail("TTL 设置正确", f"TTL={ttl}")
        else:
            results.add_fail("TTL 设置正确", "Redis 不可用")
    except Exception as e:
        results.add_fail("TTL 设置正确", str(e))
    
    # 2.4 测试空 task_id
    try:
        generator._update_status(None, TaskStatus.PENDING)
        generator._update_status("", TaskStatus.PENDING)
        results.add_pass("空 task_id 处理")
    except Exception as e:
        results.add_fail("空 task_id 处理", str(e))
    
    # 清理
    if redis_manager.available:
        redis_manager.delete(f"task:{test_task_id}")
    if status_file.exists():
        status_file.unlink()


def test_cache_operations(results):
    """测试3：缓存操作"""
    print("\n[测试组3] 缓存操作")
    
    cache_manager = RedisManager(db=1)
    
    # 3.1 测试转写结果缓存
    try:
        transcript_key = "transcript:test_bilibili_BV123_medium"
        transcript_data = {
            "language": "zh",
            "full_text": "这是测试转写文本" * 100,  # 较长文本
            "segments": [{"start": i, "end": i+1, "text": f"段落{i}"} for i in range(50)],
            "raw": {}
        }
        
        # 写入
        success = cache_manager.set(
            transcript_key,
            json.dumps(transcript_data, ensure_ascii=False),
            ttl=604800
        )
        
        # 读取
        cached = cache_manager.get(transcript_key)
        if cached:
            retrieved = json.loads(cached)
            if retrieved["full_text"] == transcript_data["full_text"]:
                results.add_pass("转写结果缓存（大数据）")
            else:
                results.add_fail("转写结果缓存（大数据）", "数据不匹配")
        else:
            results.add_fail("转写结果缓存（大数据）", "读取失败")
        
        # 清理
        cache_manager.delete(transcript_key)
    except Exception as e:
        results.add_fail("转写结果缓存（大数据）", str(e))
    
    # 3.2 测试音频元信息缓存
    try:
        audio_key = "audio:test_youtube_abc123_high"
        audio_data = {
            "file_path": "/path/to/audio.mp3",
            "title": "测试视频标题 🎵",  # 包含 emoji
            "video_id": "abc123",
            "duration": 3600,
            "raw_info": {"key": "value" * 100}  # 嵌套数据
        }
        
        success = cache_manager.set(
            audio_key,
            json.dumps(audio_data, ensure_ascii=False),
            ttl=604800
        )
        
        cached = cache_manager.get(audio_key)
        if cached:
            retrieved = json.loads(cached)
            if retrieved["title"] == audio_data["title"]:
                results.add_pass("音频元信息缓存（特殊字符）")
            else:
                results.add_fail("音频元信息缓存（特殊字符）", "数据不匹配")
        else:
            results.add_fail("音频元信息缓存（特殊字符）", "读取失败")
        
        cache_manager.delete(audio_key)
    except Exception as e:
        results.add_fail("音频元信息缓存（特殊字符）", str(e))
    
    # 3.3 测试 Markdown 缓存
    try:
        markdown_key = "markdown:test_task_md"
        markdown_content = """# 测试标题

## 章节1
这是一段很长的 Markdown 内容。
""" * 50  # 较长内容
        
        success = cache_manager.set(markdown_key, markdown_content, ttl=2592000)
        cached = cache_manager.get(markdown_key)
        
        if cached and cached == markdown_content:
            results.add_pass("Markdown 缓存（长文本）")
        else:
            results.add_fail("Markdown 缓存（长文本）", "数据不匹配")
        
        cache_manager.delete(markdown_key)
    except Exception as e:
        results.add_fail("Markdown 缓存（长文本）", str(e))
    
    # 3.4 测试缓存覆盖
    try:
        test_key = "test:overwrite"
        cache_manager.set(test_key, "value1", ttl=60)
        cache_manager.set(test_key, "value2", ttl=60)
        
        value = cache_manager.get(test_key)
        if value == "value2":
            results.add_pass("缓存覆盖")
        else:
            results.add_fail("缓存覆盖", f"期望 value2，实际 {value}")
        
        cache_manager.delete(test_key)
    except Exception as e:
        results.add_fail("缓存覆盖", str(e))
    
    # 3.5 测试批量删除
    try:
        keys = [f"test:batch_{i}" for i in range(10)]
        for key in keys:
            cache_manager.set(key, f"value_{key}", ttl=60)
        
        cache_manager.delete(*keys)
        
        # 验证删除
        all_deleted = all(cache_manager.get(key) is None for key in keys)
        if all_deleted:
            results.add_pass("批量删除")
        else:
            results.add_fail("批量删除", "部分键未删除")
    except Exception as e:
        results.add_fail("批量删除", str(e))


def test_ttl_and_expiration(results):
    """测试4：TTL 和过期机制"""
    print("\n[测试组4] TTL 和过期机制")
    
    cache_manager = RedisManager(db=1)
    
    # 4.1 测试短 TTL
    try:
        test_key = "test:short_ttl"
        cache_manager.set(test_key, "test_value", ttl=2)
        
        # 立即读取
        value1 = cache_manager.get(test_key)
        
        # 等待过期
        time.sleep(3)
        value2 = cache_manager.get(test_key)
        
        if value1 == "test_value" and value2 is None:
            results.add_pass("短 TTL 过期")
        else:
            results.add_fail("短 TTL 过期", f"value1={value1}, value2={value2}")
    except Exception as e:
        results.add_fail("短 TTL 过期", str(e))
    
    # 4.2 测试 TTL 查询
    try:
        test_key = "test:ttl_query"
        cache_manager.set(test_key, "test_value", ttl=100)
        
        ttl = cache_manager.ttl(test_key)
        if 90 <= ttl <= 100:
            results.add_pass("TTL 查询")
        else:
            results.add_fail("TTL 查询", f"TTL={ttl}")
        
        cache_manager.delete(test_key)
    except Exception as e:
        results.add_fail("TTL 查询", str(e))
    
    # 4.3 测试不存在的键
    try:
        ttl = cache_manager.ttl("test:nonexistent")
        if ttl == -2:  # -2 表示键不存在
            results.add_pass("不存在键的 TTL")
        else:
            results.add_fail("不存在键的 TTL", f"TTL={ttl}")
    except Exception as e:
        results.add_fail("不存在键的 TTL", str(e))
    
    # 4.4 测试永不过期
    try:
        test_key = "test:no_expire"
        cache_manager.set(test_key, "test_value", ttl=None)
        
        ttl = cache_manager.ttl(test_key)
        if ttl == -1:  # -1 表示永不过期
            results.add_pass("永不过期")
        else:
            results.add_fail("永不过期", f"TTL={ttl}")
        
        cache_manager.delete(test_key)
    except Exception as e:
        results.add_fail("永不过期", str(e))


def test_concurrent_operations(results):
    """测试5：并发操作"""
    print("\n[测试组5] 并发操作")
    
    cache_manager = RedisManager(db=1)
    errors = []
    
    # 5.1 并发写入
    def concurrent_write(thread_id):
        try:
            for i in range(10):
                key = f"test:concurrent_{thread_id}_{i}"
                cache_manager.set(key, f"value_{thread_id}_{i}", ttl=60)
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")
    
    try:
        threads = []
        for i in range(5):
            t = threading.Thread(target=concurrent_write, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        if not errors:
            results.add_pass("并发写入")
        else:
            results.add_fail("并发写入", f"{len(errors)} 个错误")
    except Exception as e:
        results.add_fail("并发写入", str(e))
    
    # 5.2 并发读取
    def concurrent_read(thread_id):
        try:
            for i in range(10):
                key = f"test:concurrent_0_{i}"
                cache_manager.get(key)
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")
    
    try:
        errors.clear()
        threads = []
        for i in range(5):
            t = threading.Thread(target=concurrent_read, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        if not errors:
            results.add_pass("并发读取")
        else:
            results.add_fail("并发读取", f"{len(errors)} 个错误")
    except Exception as e:
        results.add_fail("并发读取", str(e))
    
    # 清理
    try:
        keys_to_delete = []
        for i in range(5):
            for j in range(10):
                keys_to_delete.append(f"test:concurrent_{i}_{j}")
        cache_manager.delete(*keys_to_delete)
    except:
        pass


def test_edge_cases(results):
    """测试6：边界情况"""
    print("\n[测试组6] 边界情况")
    
    cache_manager = RedisManager(db=1)
    
    # 6.1 空值
    try:
        test_key = "test:empty_value"
        cache_manager.set(test_key, "", ttl=60)
        value = cache_manager.get(test_key)
        
        if value == "":
            results.add_pass("空字符串值")
        else:
            results.add_fail("空字符串值", f"期望空字符串，实际 {value}")
        
        cache_manager.delete(test_key)
    except Exception as e:
        results.add_fail("空字符串值", str(e))
    
    # 6.2 超长键名
    try:
        long_key = "test:" + "a" * 1000
        cache_manager.set(long_key, "value", ttl=60)
        value = cache_manager.get(long_key)
        
        if value == "value":
            results.add_pass("超长键名")
        else:
            results.add_fail("超长键名", "读取失败")
        
        cache_manager.delete(long_key)
    except Exception as e:
        results.add_fail("超长键名", str(e))
    
    # 6.3 超长值
    try:
        test_key = "test:long_value"
        long_value = "x" * 1000000  # 1MB
        cache_manager.set(test_key, long_value, ttl=60)
        value = cache_manager.get(test_key)
        
        if value == long_value:
            results.add_pass("超长值（1MB）")
        else:
            results.add_fail("超长值（1MB）", "数据不匹配")
        
        cache_manager.delete(test_key)
    except Exception as e:
        results.add_fail("超长值（1MB）", str(e))
    
    # 6.4 特殊字符键名
    try:
        special_keys = [
            "test:key:with:colons",
            "test:key-with-dashes",
            "test:key_with_underscores",
            "test:key.with.dots",
            "test:key/with/slashes",
        ]
        
        for key in special_keys:
            cache_manager.set(key, "value", ttl=60)
            value = cache_manager.get(key)
            if value != "value":
                results.add_fail("特殊字符键名", f"键 {key} 失败")
                break
        else:
            results.add_pass("特殊字符键名")
        
        cache_manager.delete(*special_keys)
    except Exception as e:
        results.add_fail("特殊字符键名", str(e))
    
    # 6.5 Unicode 字符
    try:
        test_key = "test:unicode"
        unicode_value = "测试 🎉 テスト 테스트 тест"
        cache_manager.set(test_key, unicode_value, ttl=60)
        value = cache_manager.get(test_key)
        
        if value == unicode_value:
            results.add_pass("Unicode 字符")
        else:
            results.add_fail("Unicode 字符", "数据不匹配")
        
        cache_manager.delete(test_key)
    except Exception as e:
        results.add_fail("Unicode 字符", str(e))


def test_hash_operations(results):
    """测试7：Hash 操作"""
    print("\n[测试组7] Hash 操作")
    
    cache_manager = RedisManager(db=0)
    
    # 7.1 基本 Hash 操作
    try:
        test_key = "test:hash_basic"
        data = {
            "field1": "value1",
            "field2": "value2",
            "field3": "value3"
        }
        
        cache_manager.hset(test_key, data, ttl=60)
        retrieved = cache_manager.hgetall(test_key)
        
        if retrieved == data:
            results.add_pass("基本 Hash 操作")
        else:
            results.add_fail("基本 Hash 操作", "数据不匹配")
        
        cache_manager.delete(test_key)
    except Exception as e:
        results.add_fail("基本 Hash 操作", str(e))
    
    # 7.2 Hash 更新
    try:
        test_key = "test:hash_update"
        data1 = {"field1": "value1"}
        data2 = {"field1": "updated", "field2": "new"}
        
        cache_manager.hset(test_key, data1, ttl=60)
        cache_manager.hset(test_key, data2, ttl=60)
        retrieved = cache_manager.hgetall(test_key)
        
        if retrieved.get("field1") == "updated" and retrieved.get("field2") == "new":
            results.add_pass("Hash 更新")
        else:
            results.add_fail("Hash 更新", f"数据不匹配: {retrieved}")
        
        cache_manager.delete(test_key)
    except Exception as e:
        results.add_fail("Hash 更新", str(e))
    
    # 7.3 空 Hash
    try:
        test_key = "test:hash_empty"
        retrieved = cache_manager.hgetall(test_key)
        
        if retrieved is None:
            results.add_pass("空 Hash")
        else:
            results.add_fail("空 Hash", f"期望 None，实际 {retrieved}")
    except Exception as e:
        results.add_fail("空 Hash", str(e))


def test_error_handling(results):
    """测试8：错误处理"""
    print("\n[测试组8] 错误处理")
    
    # 8.1 测试 Redis 不可用时的降级
    try:
        # 模拟 Redis 不可用（通过重置连接并设置错误的配置）
        original_host = os.getenv("REDIS_HOST")
        os.environ["REDIS_HOST"] = "invalid_host_12345"
        
        RedisClient.reset()
        manager = RedisManager(db=0)
        
        if not manager.available:
            results.add_pass("Redis 不可用检测")
        else:
            results.add_fail("Redis 不可用检测", "应该检测到不可用")
        
        # 恢复配置
        os.environ["REDIS_HOST"] = original_host
        RedisClient.reset()
    except Exception as e:
        results.add_fail("Redis 不可用检测", str(e))
        # 确保恢复配置
        os.environ["REDIS_HOST"] = original_host
        RedisClient.reset()
    
    # 8.2 测试操作失败时的返回值
    try:
        manager = RedisManager(db=0)
        if manager.available:
            # 测试删除不存在的键
            result = manager.delete("test:nonexistent_key_12345")
            results.add_pass("删除不存在的键")
        else:
            results.add_fail("删除不存在的键", "Redis 不可用")
    except Exception as e:
        results.add_fail("删除不存在的键", str(e))


def test_redis_info(results):
    """测试9：Redis 信息查询"""
    print("\n[测试组9] Redis 信息查询")
    
    try:
        client = RedisClient.get_instance(db=0)
        if client:
            info = client.info()
            
            # 验证关键信息
            required_keys = [
                "redis_version",
                "used_memory_human",
                "connected_clients",
                "total_commands_processed",
                "uptime_in_days"
            ]
            
            missing_keys = [key for key in required_keys if key not in info]
            
            if not missing_keys:
                results.add_pass("Redis 信息查询")
            else:
                results.add_fail("Redis 信息查询", f"缺少键: {missing_keys}")
        else:
            results.add_fail("Redis 信息查询", "无法获取客户端")
    except Exception as e:
        results.add_fail("Redis 信息查询", str(e))


def main():
    """主测试函数"""
    print("=" * 70)
    print("Redis 集成综合测试")
    print("=" * 70)
    
    results = TestResults()
    
    # 执行所有测试组
    test_groups = [
        ("Redis 连接", test_redis_connection),
        ("任务状态管理", test_task_status_management),
        ("缓存操作", test_cache_operations),
        ("TTL 和过期", test_ttl_and_expiration),
        ("并发操作", test_concurrent_operations),
        ("边界情况", test_edge_cases),
        ("Hash 操作", test_hash_operations),
        ("错误处理", test_error_handling),
        ("Redis 信息", test_redis_info),
    ]
    
    for group_name, test_func in test_groups:
        try:
            test_func(results)
        except Exception as e:
            print(f"\n❌ 测试组 [{group_name}] 执行失败: {e}")
            results.add_fail(f"测试组 {group_name}", str(e))
    
    # 输出总结
    success = results.summary()
    
    if success:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误详情")
        return 1


if __name__ == "__main__":
    sys.exit(main())
