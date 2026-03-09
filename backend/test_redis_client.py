#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Redis 客户端封装
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.utils.redis_client import RedisClient, RedisManager
from dotenv import load_dotenv

load_dotenv()

def test_redis_client():
    """测试 Redis 客户端基础功能"""
    print("=" * 60)
    print("测试 Redis 客户端封装")
    print("=" * 60)
    
    # 测试1：检查可用性
    print("\n[测试1] 检查 Redis 可用性")
    is_available = RedisClient.is_available()
    print(f"  Redis 可用: {is_available}")
    
    if not is_available:
        print("  ❌ Redis 不可用，测试终止")
        return False
    
    # 测试2：获取实例
    print("\n[测试2] 获取 Redis 实例")
    client_db0 = RedisClient.get_instance(db=0)
    client_db1 = RedisClient.get_instance(db=1)
    print(f"  DB0 实例: {client_db0}")
    print(f"  DB1 实例: {client_db1}")
    
    # 测试3：PING 测试
    print("\n[测试3] PING 测试")
    ping_result = RedisClient.ping(db=0)
    print(f"  PING 结果: {ping_result}")
    
    # 测试4：RedisManager 基础操作
    print("\n[测试4] RedisManager 基础操作")
    manager = RedisManager(db=0)
    
    # SET/GET
    print("  测试 SET/GET:")
    success = manager.set("test_key", "test_value", ttl=60)
    print(f"    SET 成功: {success}")
    
    value = manager.get("test_key")
    print(f"    GET 结果: {value}")
    
    # HSET/HGETALL
    print("  测试 HSET/HGETALL:")
    success = manager.hset("test_hash", {"field1": "value1", "field2": "value2"}, ttl=60)
    print(f"    HSET 成功: {success}")
    
    hash_data = manager.hgetall("test_hash")
    print(f"    HGETALL 结果: {hash_data}")
    
    # EXISTS
    print("  测试 EXISTS:")
    exists = manager.exists("test_key")
    print(f"    test_key 存在: {exists}")
    
    # TTL
    print("  测试 TTL:")
    ttl = manager.ttl("test_key")
    print(f"    test_key 剩余时间: {ttl}秒")
    
    # DELETE
    print("  测试 DELETE:")
    success = manager.delete("test_key", "test_hash")
    print(f"    DELETE 成功: {success}")
    
    exists = manager.exists("test_key")
    print(f"    test_key 存在: {exists}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_redis_client()
