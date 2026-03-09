#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest 配置文件
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from dotenv import load_dotenv
from app.utils.redis_client import RedisManager, RedisClient

load_dotenv()


@pytest.fixture(scope="session")
def redis_manager():
    """Redis 管理器 fixture（会话级别）"""
    manager = RedisManager(db=1)
    if not manager.available:
        pytest.fail("Redis 不可用，测试失败！请确保 Redis 服务正在运行")
    yield manager
    # 清理测试数据
    try:
        keys = manager.client.keys("test:*")
        if keys:
            manager.client.delete(*keys)
    except:
        pass


@pytest.fixture(scope="function")
def clean_redis(redis_manager):
    """每个测试前清理 Redis（函数级别）"""
    # 测试前清理
    try:
        keys = redis_manager.client.keys("test:*")
        if keys:
            redis_manager.client.delete(*keys)
    except:
        pass
    
    yield redis_manager
    
    # 测试后清理
    try:
        keys = redis_manager.client.keys("test:*")
        if keys:
            redis_manager.client.delete(*keys)
    except:
        pass


@pytest.fixture(scope="session")
def redis_db0():
    """Redis DB0 管理器（用于任务状态测试）"""
    manager = RedisManager(db=0)
    if not manager.available:
        pytest.fail("Redis 不可用，测试失败！请确保 Redis 服务正在运行")
    yield manager
    # 清理测试数据
    try:
        keys = manager.client.keys("test:*")
        if keys:
            manager.client.delete(*keys)
    except:
        pass
