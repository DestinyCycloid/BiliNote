#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 任务状态管理测试
"""
import pytest
import time
import json


class TestTaskStatusManagement:
    """任务状态管理测试"""
    
    def test_task_status_create(self, redis_db0):
        """测试创建任务状态"""
        task_id = "test_task_001"
        redis_key = f"test:task:{task_id}"
        
        status_data = {
            "status": "PENDING",
            "message": "任务已创建",
            "updated_at": str(time.time())
        }
        
        success = redis_db0.hset(redis_key, status_data, ttl=86400)
        assert success is True
        
        # 验证数据
        result = redis_db0.hgetall(redis_key)
        assert result is not None
        assert result["status"] == "PENDING"
        assert result["message"] == "任务已创建"
        
        # 清理
        redis_db0.delete(redis_key)
    
    def test_task_status_update(self, redis_db0):
        """测试更新任务状态"""
        task_id = "test_task_002"
        redis_key = f"test:task:{task_id}"
        
        # 创建初始状态
        status_data = {
            "status": "PENDING",
            "message": "任务已创建",
            "updated_at": str(time.time())
        }
        redis_db0.hset(redis_key, status_data, ttl=86400)
        
        # 更新状态
        time.sleep(0.1)
        updated_data = {
            "status": "DOWNLOADING",
            "message": "正在下载音频",
            "updated_at": str(time.time())
        }
        redis_db0.hset(redis_key, updated_data, ttl=86400)
        
        # 验证更新
        result = redis_db0.hgetall(redis_key)
        assert result["status"] == "DOWNLOADING"
        assert result["message"] == "正在下载音频"
        
        # 清理
        redis_db0.delete(redis_key)
    
    def test_task_status_lifecycle(self, redis_db0):
        """测试任务状态生命周期"""
        task_id = "test_task_003"
        redis_key = f"test:task:{task_id}"
        
        statuses = [
            ("PENDING", "任务已创建"),
            ("PARSING", "正在解析链接"),
            ("DOWNLOADING", "正在下载音频"),
            ("TRANSCRIBING", "正在转写音频"),
            ("SUMMARIZING", "正在生成笔记"),
            ("SAVING", "正在保存结果"),
            ("SUCCESS", "任务完成")
        ]
        
        for status, message in statuses:
            status_data = {
                "status": status,
                "message": message,
                "updated_at": str(time.time())
            }
            redis_db0.hset(redis_key, status_data, ttl=86400)
            
            # 验证每次更新
            result = redis_db0.hgetall(redis_key)
            assert result["status"] == status
            assert result["message"] == message
            
            time.sleep(0.05)
        
        # 清理
        redis_db0.delete(redis_key)
    
    def test_task_status_ttl(self, redis_db0):
        """测试任务状态 TTL"""
        task_id = "test_task_004"
        redis_key = f"test:task:{task_id}"
        
        status_data = {
            "status": "PENDING",
            "message": "测试 TTL",
            "updated_at": str(time.time())
        }
        
        # 设置 24 小时 TTL
        redis_db0.hset(redis_key, status_data, ttl=86400)
        
        # 检查 TTL
        ttl = redis_db0.ttl(redis_key)
        assert 86300 < ttl <= 86400
        
        # 清理
        redis_db0.delete(redis_key)
    
    def test_task_status_query_performance(self, redis_db0):
        """测试任务状态查询性能"""
        task_id = "test_task_005"
        redis_key = f"test:task:{task_id}"
        
        status_data = {
            "status": "TRANSCRIBING",
            "message": "正在转写",
            "progress": "50%",
            "updated_at": str(time.time())
        }
        redis_db0.hset(redis_key, status_data, ttl=86400)
        
        # 测试 100 次查询
        start_time = time.time()
        for _ in range(100):
            redis_db0.hgetall(redis_key)
        elapsed = time.time() - start_time
        
        # 平均每次查询应该小于 5ms
        avg_time = elapsed / 100
        assert avg_time < 0.005
        
        # 清理
        redis_db0.delete(redis_key)


class TestPlaylistTaskStatus:
    """合集任务状态测试"""
    
    def test_playlist_progress_update(self, redis_db0):
        """测试合集处理进度更新"""
        task_id = "test_playlist_001"
        redis_key = f"test:task:{task_id}"
        
        total_videos = 10
        
        # 模拟处理进度
        for i in range(1, total_videos + 1):
            status_data = {
                "status": "TRANSCRIBING",
                "message": f"正在处理合集：已转写 {i}/{total_videos}",
                "progress": f"{i}/{total_videos}",
                "updated_at": str(time.time())
            }
            redis_db0.hset(redis_key, status_data, ttl=86400)
            
            # 验证进度
            result = redis_db0.hgetall(redis_key)
            assert result["progress"] == f"{i}/{total_videos}"
            
            time.sleep(0.01)
        
        # 验证最终状态
        result = redis_db0.hgetall(redis_key)
        assert result["progress"] == f"{total_videos}/{total_videos}"
        
        # 清理
        redis_db0.delete(redis_key)
    
    def test_playlist_realtime_progress(self, redis_db0):
        """测试合集实时进度查询"""
        task_id = "test_playlist_002"
        redis_key = f"test:task:{task_id}"
        
        # 初始状态
        status_data = {
            "status": "TRANSCRIBING",
            "message": "正在处理合集",
            "transcribed": "0",
            "summarized": "0",
            "total": "5",
            "updated_at": str(time.time())
        }
        redis_db0.hset(redis_key, status_data, ttl=86400)
        
        # 模拟并行处理
        updates = [
            {"transcribed": "1", "summarized": "0"},
            {"transcribed": "2", "summarized": "1"},
            {"transcribed": "3", "summarized": "2"},
            {"transcribed": "5", "summarized": "3"},
            {"transcribed": "5", "summarized": "5"},
        ]
        
        for update in updates:
            status_data.update(update)
            status_data["updated_at"] = str(time.time())
            redis_db0.hset(redis_key, status_data, ttl=86400)
            
            # 验证更新
            result = redis_db0.hgetall(redis_key)
            assert result["transcribed"] == update["transcribed"]
            assert result["summarized"] == update["summarized"]
            
            time.sleep(0.01)
        
        # 清理
        redis_db0.delete(redis_key)


class TestConcurrentTaskStatus:
    """并发任务状态测试"""
    
    def test_multiple_tasks_concurrent(self, redis_db0):
        """测试多个任务并发状态管理"""
        task_count = 10
        
        # 创建多个任务
        for i in range(task_count):
            task_id = f"test_concurrent_{i}"
            redis_key = f"test:task:{task_id}"
            
            status_data = {
                "status": "TRANSCRIBING",
                "message": f"任务 {i} 正在处理",
                "updated_at": str(time.time())
            }
            redis_db0.hset(redis_key, status_data, ttl=86400)
        
        # 验证所有任务都创建成功
        for i in range(task_count):
            task_id = f"test_concurrent_{i}"
            redis_key = f"test:task:{task_id}"
            
            result = redis_db0.hgetall(redis_key)
            assert result is not None
            assert result["status"] == "TRANSCRIBING"
            assert f"任务 {i}" in result["message"]
        
        # 清理
        for i in range(task_count):
            redis_db0.delete(f"test:task:test_concurrent_{i}")
    
    def test_high_frequency_updates(self, redis_db0):
        """测试高频率状态更新"""
        task_id = "test_high_freq"
        redis_key = f"test:task:{task_id}"
        
        # 100 次快速更新
        start_time = time.time()
        for i in range(100):
            status_data = {
                "status": "TRANSCRIBING",
                "message": f"进度 {i}%",
                "updated_at": str(time.time())
            }
            redis_db0.hset(redis_key, status_data, ttl=86400)
        elapsed = time.time() - start_time
        
        # 验证最终状态
        result = redis_db0.hgetall(redis_key)
        assert result["message"] == "进度 99%"
        
        # 平均每次更新应该小于 5ms
        avg_time = elapsed / 100
        assert avg_time < 0.005
        
        # 清理
        redis_db0.delete(redis_key)
