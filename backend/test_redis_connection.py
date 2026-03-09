#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 连接测试脚本
"""
import redis
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_redis_connection():
    """测试 Redis 连接"""
    try:
        # 从环境变量读取配置
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_password = os.getenv("REDIS_PASSWORD", None) or None
        
        print(f"正在连接 Redis: {redis_host}:{redis_port}")
        
        # 创建 Redis 客户端
        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # 测试 PING
        response = client.ping()
        print(f"✅ PING 测试成功: {response}")
        
        # 测试写入
        client.set("test_key", "Hello Redis!")
        print("✅ 写入测试成功")
        
        # 测试读取
        value = client.get("test_key")
        print(f"✅ 读取测试成功: {value}")
        
        # 测试删除
        client.delete("test_key")
        print("✅ 删除测试成功")
        
        # 获取 Redis 信息
        info = client.info()
        print(f"\n📊 Redis 服务器信息:")
        print(f"  - 版本: {info.get('redis_version')}")
        print(f"  - 运行模式: {info.get('redis_mode')}")
        print(f"  - 已用内存: {info.get('used_memory_human')}")
        print(f"  - 连接客户端数: {info.get('connected_clients')}")
        print(f"  - 运行天数: {info.get('uptime_in_days')}")
        
        print("\n🎉 所有测试通过！Redis 连接正常")
        return True
        
    except redis.ConnectionError as e:
        print(f"❌ 连接失败: {e}")
        print("\n请检查:")
        print("  1. Redis 服务是否启动")
        print("  2. 防火墙是否开放 6379 端口")
        print("  3. Redis 配置是否允许远程连接 (bind 0.0.0.0)")
        return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_redis_connection()
