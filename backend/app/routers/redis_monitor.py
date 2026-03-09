#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 监控接口
提供 Redis 状态查询、缓存统计等功能
"""
from fastapi import APIRouter
from app.utils.redis_client import RedisClient, RedisManager
from app.utils.response import ResponseWrapper as R
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/redis/info")
def get_redis_info():
    """
    获取 Redis 服务器信息
    
    返回：
    - connected: Redis 连接状态
    - version: Redis 版本
    - used_memory: 已用内存
    - connected_clients: 连接客户端数
    - total_commands: 总命令数
    - uptime_days: 运行天数
    """
    try:
        # 检查 Redis 是否可用
        if not RedisClient.is_available():
            return R.error("Redis 不可用，系统已降级到文件系统模式", code=503)
        
        # 获取 Redis 客户端
        client = RedisClient.get_instance(db=0)
        if not client:
            return R.error("无法连接到 Redis", code=503)
        
        # 获取 Redis 信息
        info = client.info()
        
        return R.success({
            "connected": True,
            "version": info.get("redis_version"),
            "mode": info.get("redis_mode"),
            "used_memory": info.get("used_memory_human"),
            "used_memory_peak": info.get("used_memory_peak_human"),
            "connected_clients": info.get("connected_clients"),
            "total_commands": info.get("total_commands_processed"),
            "uptime_days": info.get("uptime_in_days"),
            "uptime_seconds": info.get("uptime_in_seconds"),
        })
        
    except Exception as e:
        logger.error(f"获取 Redis 信息失败: {e}")
        return R.error(f"获取 Redis 信息失败: {str(e)}", code=500)


@router.get("/redis/stats")
def get_redis_stats():
    """
    获取 Redis 缓存统计信息
    
    返回各个数据库的键数量统计
    """
    try:
        if not RedisClient.is_available():
            return R.error("Redis 不可用", code=503)
        
        stats = {}
        
        # DB0: 任务状态
        db0_client = RedisClient.get_instance(db=0)
        if db0_client:
            task_keys = db0_client.keys("task:*")
            stats["db0_tasks"] = {
                "description": "任务状态",
                "count": len(task_keys),
                "ttl": "24小时"
            }
        
        # DB1: 缓存
        db1_client = RedisClient.get_instance(db=1)
        if db1_client:
            transcript_keys = db1_client.keys("transcript:*")
            audio_keys = db1_client.keys("audio:*")
            markdown_keys = db1_client.keys("markdown:*")
            
            stats["db1_cache"] = {
                "description": "缓存数据",
                "transcript_count": len(transcript_keys),
                "audio_count": len(audio_keys),
                "markdown_count": len(markdown_keys),
                "total_count": len(transcript_keys) + len(audio_keys) + len(markdown_keys),
                "ttl": "转写/音频: 7天, Markdown: 30天"
            }
        
        return R.success(stats)
        
    except Exception as e:
        logger.error(f"获取 Redis 统计信息失败: {e}")
        return R.error(f"获取统计信息失败: {str(e)}", code=500)


@router.post("/redis/clear")
def clear_redis_cache(data: dict):
    """
    清理 Redis 缓存
    
    参数：
    - cache_type: 缓存类型 ("tasks" | "transcripts" | "audio" | "markdown" | "all")
    """
    try:
        if not RedisClient.is_available():
            return R.error("Redis 不可用", code=503)
        
        cache_type = data.get("cache_type", "all")
        deleted_count = 0
        
        if cache_type in ["tasks", "all"]:
            # 清理任务状态（DB0）
            db0_client = RedisClient.get_instance(db=0)
            if db0_client:
                task_keys = db0_client.keys("task:*")
                if task_keys:
                    deleted_count += db0_client.delete(*task_keys)
                logger.info(f"已清理 {len(task_keys)} 个任务状态")
        
        if cache_type in ["transcripts", "all"]:
            # 清理转写缓存（DB1）
            db1_client = RedisClient.get_instance(db=1)
            if db1_client:
                transcript_keys = db1_client.keys("transcript:*")
                if transcript_keys:
                    deleted_count += db1_client.delete(*transcript_keys)
                logger.info(f"已清理 {len(transcript_keys)} 个转写缓存")
        
        if cache_type in ["audio", "all"]:
            # 清理音频缓存（DB1）
            db1_client = RedisClient.get_instance(db=1)
            if db1_client:
                audio_keys = db1_client.keys("audio:*")
                if audio_keys:
                    deleted_count += db1_client.delete(*audio_keys)
                logger.info(f"已清理 {len(audio_keys)} 个音频缓存")
        
        if cache_type in ["markdown", "all"]:
            # 清理 Markdown 缓存（DB1）
            db1_client = RedisClient.get_instance(db=1)
            if db1_client:
                markdown_keys = db1_client.keys("markdown:*")
                if markdown_keys:
                    deleted_count += db1_client.delete(*markdown_keys)
                logger.info(f"已清理 {len(markdown_keys)} 个 Markdown 缓存")
        
        return R.success({
            "message": f"成功清理 {deleted_count} 个缓存项",
            "deleted_count": deleted_count,
            "cache_type": cache_type
        })
        
    except Exception as e:
        logger.error(f"清理 Redis 缓存失败: {e}")
        return R.error(f"清理缓存失败: {str(e)}", code=500)


@router.get("/redis/ping")
def ping_redis():
    """
    测试 Redis 连接
    """
    try:
        is_available = RedisClient.ping(db=0)
        
        if is_available:
            return R.success({
                "connected": True,
                "message": "Redis 连接正常"
            })
        else:
            return R.error("Redis 连接失败，系统已降级到文件系统模式", code=503)
            
    except Exception as e:
        logger.error(f"Redis PING 失败: {e}")
        return R.error(f"连接测试失败: {str(e)}", code=500)
