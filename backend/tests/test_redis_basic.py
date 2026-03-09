#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 基础功能测试
"""
import pytest
import time
from app.utils.redis_client import RedisManager, RedisClient


class TestRedisConnection:
    """Redis 连接测试"""
    
    def test_redis_available(self, redis_manager):
        """测试 Redis 是否可用"""
        assert redis_manager.available is True
        assert redis_manager.client is not None
    
    def test_redis_ping(self, redis_manager):
        """测试 Redis PING"""
        assert redis_manager.client.ping() is True
    
    def test_redis_client_singleton(self):
        """测试 Redis 客户端单例模式"""
        client1 = RedisClient.get_instance(db=1)
        client2 = RedisClient.get_instance(db=1)
        assert client1 is client2
    
    def test_redis_multi_db(self):
        """测试多 DB 支持"""
        client_db0 = RedisClient.get_instance(db=0)
        client_db1 = RedisClient.get_instance(db=1)
        assert client_db0 is not None
        assert client_db1 is not None
        assert client_db0 is not client_db1


class TestRedisBasicOperations:
    """Redis 基础操作测试"""
    
    def test_set_and_get(self, clean_redis):
        """测试 SET 和 GET"""
        key = "test:basic:key1"
        value = "test_value"
        
        assert clean_redis.set(key, value) is True
        assert clean_redis.get(key) == value
    
    def test_set_with_ttl(self, clean_redis):
        """测试带 TTL 的 SET"""
        key = "test:basic:key2"
        value = "test_value"
        ttl = 2
        
        assert clean_redis.set(key, value, ttl=ttl) is True
        assert clean_redis.get(key) == value
        
        # 等待过期
        time.sleep(ttl + 0.5)
        assert clean_redis.get(key) is None
    
    def test_delete(self, clean_redis):
        """测试 DELETE"""
        key = "test:basic:key3"
        value = "test_value"
        
        clean_redis.set(key, value)
        assert clean_redis.exists(key) is True
        
        assert clean_redis.delete(key) is True
        assert clean_redis.exists(key) is False
    
    def test_exists(self, clean_redis):
        """测试 EXISTS"""
        key = "test:basic:key4"
        
        assert clean_redis.exists(key) is False
        
        clean_redis.set(key, "value")
        assert clean_redis.exists(key) is True
    
    def test_ttl(self, clean_redis):
        """测试 TTL"""
        key = "test:basic:key5"
        ttl = 10
        
        clean_redis.set(key, "value", ttl=ttl)
        
        remaining_ttl = clean_redis.ttl(key)
        assert 0 < remaining_ttl <= ttl


class TestRedisHashOperations:
    """Redis Hash 操作测试"""
    
    def test_hset_and_hgetall(self, clean_redis):
        """测试 HSET 和 HGETALL"""
        key = "test:hash:key1"
        data = {
            "field1": "value1",
            "field2": "value2",
            "field3": "value3"
        }
        
        assert clean_redis.hset(key, data) is True
        result = clean_redis.hgetall(key)
        assert result == data
    
    def test_hset_with_ttl(self, clean_redis):
        """测试带 TTL 的 HSET"""
        key = "test:hash:key2"
        data = {"field": "value"}
        ttl = 2
        
        assert clean_redis.hset(key, data, ttl=ttl) is True
        assert clean_redis.hgetall(key) == data
        
        # 等待过期
        time.sleep(ttl + 0.5)
        assert clean_redis.hgetall(key) is None
    
    def test_hash_update(self, clean_redis):
        """测试 Hash 更新"""
        key = "test:hash:key3"
        
        # 初始数据
        data1 = {"field1": "value1"}
        clean_redis.hset(key, data1)
        
        # 更新数据
        data2 = {"field1": "updated", "field2": "new"}
        clean_redis.hset(key, data2)
        
        result = clean_redis.hgetall(key)
        assert result["field1"] == "updated"
        assert result["field2"] == "new"


class TestRedisEdgeCases:
    """Redis 边界情况测试"""
    
    def test_empty_value(self, clean_redis):
        """测试空值"""
        key = "test:edge:empty"
        
        assert clean_redis.set(key, "") is True
        assert clean_redis.get(key) == ""
    
    def test_large_value(self, clean_redis):
        """测试大值"""
        key = "test:edge:large"
        value = "x" * 10000  # 10KB
        
        assert clean_redis.set(key, value) is True
        assert clean_redis.get(key) == value
    
    def test_special_characters(self, clean_redis):
        """测试特殊字符"""
        key = "test:edge:special"
        value = "测试\n换行\t制表符\"引号'单引号"
        
        assert clean_redis.set(key, value) is True
        assert clean_redis.get(key) == value
    
    def test_unicode(self, clean_redis):
        """测试 Unicode"""
        key = "test:edge:unicode"
        value = "中文测试 🎉 emoji"
        
        assert clean_redis.set(key, value) is True
        assert clean_redis.get(key) == value
    
    def test_nonexistent_key(self, clean_redis):
        """测试不存在的键"""
        key = "test:edge:nonexistent"
        
        assert clean_redis.get(key) is None
        assert clean_redis.hgetall(key) is None
        assert clean_redis.exists(key) is False
        assert clean_redis.ttl(key) == -2
