#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 客户端管理模块
提供单例模式的 Redis 连接管理，支持多 DB 切换和自动降级
"""
import os
import redis
from typing import Optional
from dotenv import load_dotenv
from app.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class RedisClient:
    """Redis 客户端单例管理类"""
    
    _instances = {}  # 存储不同 DB 的实例
    _available = None  # Redis 可用性标志
    
    def __init__(self):
        """不应直接实例化，请使用 get_instance() 方法"""
        raise RuntimeError("请使用 RedisClient.get_instance() 获取实例")
    
    @staticmethod
    def get_db_from_env(db_type: str) -> int:
        """
        从环境变量获取 Redis DB 编号
        
        :param db_type: DB 类型，可选值: 'task', 'cache', 'queue'
        :return: DB 编号
        """
        db_map = {
            'task': int(os.getenv('REDIS_DB_TASK', 0)),
            'cache': int(os.getenv('REDIS_DB_CACHE', 1)),
            'queue': int(os.getenv('REDIS_DB_QUEUE', 2)),
        }
        return db_map.get(db_type, 0)
    
    @classmethod
    def get_instance(cls, db: int = 0) -> Optional[redis.Redis]:
        """
        获取 Redis 客户端实例（单例模式）
        
        :param db: Redis 数据库编号（0-15）
        :return: Redis 客户端实例，连接失败时返回 None
        """
        # 如果已经检测到 Redis 不可用，直接返回 None
        if cls._available is False:
            return None
        
        # 如果该 DB 的实例已存在，直接返回
        if db in cls._instances:
            return cls._instances[db]
        
        # 创建新实例
        try:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", 6379))
            password = os.getenv("REDIS_PASSWORD", None) or None
            
            # 如果没有指定 db，使用默认值
            if db is None:
                db = 0
            
            logger.info(f"正在连接 Redis: {host}:{port} DB={db}")
            
            client = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                decode_responses=True,  # 自动解码为字符串
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,  # 每30秒检查连接健康
            )
            
            # 测试连接
            client.ping()
            
            # 缓存实例
            cls._instances[db] = client
            cls._available = True
            
            logger.info(f"✅ Redis 连接成功: {host}:{port} DB={db}")
            return client
            
        except redis.ConnectionError as e:
            logger.warning(f"❌ Redis 连接失败: {e}")
            logger.warning("系统将降级到文件系统模式")
            cls._available = False
            return None
            
        except Exception as e:
            logger.error(f"❌ Redis 初始化异常: {e}")
            cls._available = False
            return None
    
    @classmethod
    def is_available(cls) -> bool:
        """
        检查 Redis 是否可用
        
        :return: True 表示可用，False 表示不可用
        """
        if cls._available is None:
            # 首次检查，尝试连接
            client = cls.get_instance(db=0)
            return client is not None
        return cls._available
    
    @classmethod
    def ping(cls, db: int = 0) -> bool:
        """
        测试 Redis 连接
        
        :param db: 数据库编号
        :return: True 表示连接正常，False 表示连接失败
        """
        try:
            client = cls.get_instance(db=db)
            if client is None:
                return False
            return client.ping()
        except:
            cls._available = False
            return False
    
    @classmethod
    def reset(cls):
        """
        重置所有连接（用于测试或重新连接）
        """
        for client in cls._instances.values():
            try:
                client.close()
            except:
                pass
        cls._instances.clear()
        cls._available = None
        logger.info("Redis 连接已重置")


class RedisManager:
    """Redis 操作管理类，提供高级封装"""
    
    def __init__(self, db: int = 0):
        """
        初始化 Redis 管理器
        
        :param db: Redis 数据库编号
        """
        self.db = db
        self.client = RedisClient.get_instance(db=db)
        self.available = self.client is not None
    
    @classmethod
    def for_task(cls):
        """创建用于任务状态管理的 Redis 管理器"""
        db = RedisClient.get_db_from_env('task')
        return cls(db=db)
    
    @classmethod
    def for_cache(cls):
        """创建用于缓存的 Redis 管理器"""
        db = RedisClient.get_db_from_env('cache')
        return cls(db=db)
    
    @classmethod
    def for_queue(cls):
        """创建用于任务队列的 Redis 管理器"""
        db = RedisClient.get_db_from_env('queue')
        return cls(db=db)
    
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """
        设置键值对
        
        :param key: 键
        :param value: 值
        :param ttl: 过期时间（秒），None 表示永不过期
        :return: 成功返回 True，失败返回 False
        """
        if not self.available:
            return False
        
        try:
            if ttl:
                self.client.setex(key, ttl, value)
            else:
                self.client.set(key, value)
            return True
        except Exception as e:
            logger.warning(f"Redis SET 失败: {e}")
            return False
    
    def get(self, key: str) -> Optional[str]:
        """
        获取键对应的值
        
        :param key: 键
        :return: 值，不存在或失败时返回 None
        """
        if not self.available:
            return None
        
        try:
            return self.client.get(key)
        except Exception as e:
            logger.warning(f"Redis GET 失败: {e}")
            return None
    
    def hset(self, key: str, mapping: dict, ttl: Optional[int] = None) -> bool:
        """
        设置 Hash 类型数据
        
        :param key: 键
        :param mapping: 字段-值映射字典
        :param ttl: 过期时间（秒）
        :return: 成功返回 True，失败返回 False
        """
        if not self.available:
            return False
        
        try:
            self.client.hset(key, mapping=mapping)
            if ttl:
                self.client.expire(key, ttl)
            return True
        except Exception as e:
            logger.warning(f"Redis HSET 失败: {e}")
            return False
    
    def hgetall(self, key: str) -> Optional[dict]:
        """
        获取 Hash 类型的所有字段
        
        :param key: 键
        :return: 字段-值映射字典，不存在或失败时返回 None
        """
        if not self.available:
            return None
        
        try:
            data = self.client.hgetall(key)
            return data if data else None
        except Exception as e:
            logger.warning(f"Redis HGETALL 失败: {e}")
            return None
    
    def delete(self, *keys: str) -> bool:
        """
        删除一个或多个键
        
        :param keys: 要删除的键
        :return: 成功返回 True，失败返回 False
        """
        if not self.available:
            return False
        
        try:
            self.client.delete(*keys)
            return True
        except Exception as e:
            logger.warning(f"Redis DELETE 失败: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在
        
        :param key: 键
        :return: 存在返回 True，不存在或失败返回 False
        """
        if not self.available:
            return False
        
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Redis EXISTS 失败: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """
        获取键的剩余过期时间
        
        :param key: 键
        :return: 剩余秒数，-1 表示永不过期，-2 表示不存在
        """
        if not self.available:
            return -2
        
        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.warning(f"Redis TTL 失败: {e}")
            return -2
