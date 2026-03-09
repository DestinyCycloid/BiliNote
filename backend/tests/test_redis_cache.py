#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 缓存功能测试
"""
import pytest
import json
import time
from dataclasses import asdict
from app.models.transcriber_model import TranscriptResult, TranscriptSegment
from app.models.audio_model import AudioDownloadResult


class TestTranscriptCache:
    """转写结果缓存测试"""
    
    def test_transcript_cache_write_and_read(self, clean_redis):
        """测试转写缓存的写入和读取"""
        cache_key = "test:transcript:video123"
        
        # 创建转写结果
        transcript = TranscriptResult(
            language="zh",
            full_text="这是测试文本",
            segments=[
                TranscriptSegment(start=0.0, end=2.0, text="这是测试"),
                TranscriptSegment(start=2.0, end=4.0, text="文本")
            ],
            raw={}
        )
        
        # 写入缓存
        transcript_data = asdict(transcript)
        success = clean_redis.set(
            cache_key,
            json.dumps(transcript_data, ensure_ascii=False),
            ttl=604800
        )
        assert success is True
        
        # 读取缓存
        cached_data = clean_redis.get(cache_key)
        assert cached_data is not None
        
        data = json.loads(cached_data)
        assert data["language"] == "zh"
        assert data["full_text"] == "这是测试文本"
        assert len(data["segments"]) == 2
    
    def test_transcript_cache_ttl(self, clean_redis):
        """测试转写缓存的 TTL"""
        cache_key = "test:transcript:video456"
        
        transcript = TranscriptResult(
            language="zh",
            full_text="测试 TTL",
            segments=[],
            raw={}
        )
        
        # 写入缓存（7天）
        clean_redis.set(
            cache_key,
            json.dumps(asdict(transcript), ensure_ascii=False),
            ttl=604800
        )
        
        # 检查 TTL
        ttl = clean_redis.ttl(cache_key)
        assert 604700 < ttl <= 604800  # 允许一些误差
    
    def test_transcript_cache_miss(self, clean_redis):
        """测试转写缓存未命中"""
        cache_key = "test:transcript:nonexistent"
        
        cached_data = clean_redis.get(cache_key)
        assert cached_data is None


class TestAudioCache:
    """音频元信息缓存测试"""
    
    def test_audio_cache_write_and_read(self, clean_redis):
        """测试音频缓存的写入和读取"""
        cache_key = "test:audio:bilibili_BV123_high"
        
        # 创建音频元信息
        audio = AudioDownloadResult(
            file_path="/path/to/audio.mp3",
            title="测试视频",
            duration=300.0,
            cover_url="https://example.com/cover.jpg",
            platform="bilibili",
            video_id="BV123",
            raw_info={"quality": "high"}
        )
        
        # 写入缓存
        audio_data = asdict(audio)
        success = clean_redis.set(
            cache_key,
            json.dumps(audio_data, ensure_ascii=False),
            ttl=604800
        )
        assert success is True
        
        # 读取缓存
        cached_data = clean_redis.get(cache_key)
        assert cached_data is not None
        
        data = json.loads(cached_data)
        assert data["title"] == "测试视频"
        assert data["platform"] == "bilibili"
        assert data["video_id"] == "BV123"


class TestMarkdownCache:
    """Markdown 结果缓存测试"""
    
    def test_markdown_cache_write_and_read(self, clean_redis):
        """测试 Markdown 缓存的写入和读取"""
        cache_key = "test:markdown:task123"
        markdown = "# 测试标题\n\n这是测试内容"
        
        # 写入缓存（30天）
        success = clean_redis.set(cache_key, markdown, ttl=2592000)
        assert success is True
        
        # 读取缓存
        cached_markdown = clean_redis.get(cache_key)
        assert cached_markdown == markdown
    
    def test_markdown_cache_ttl(self, clean_redis):
        """测试 Markdown 缓存的 TTL"""
        cache_key = "test:markdown:task456"
        markdown = "# 测试"
        
        # 写入缓存（30天）
        clean_redis.set(cache_key, markdown, ttl=2592000)
        
        # 检查 TTL
        ttl = clean_redis.ttl(cache_key)
        assert 2591900 < ttl <= 2592000


class TestPlaylistCache:
    """合集处理缓存测试"""
    
    def test_playlist_transcript_cache(self, clean_redis):
        """测试合集转写缓存"""
        # 模拟 3 个视频
        for i in range(3):
            cache_key = f"test:transcript:playlist_video{i}"
            
            transcript = TranscriptResult(
                language="zh",
                full_text=f"这是视频 {i} 的转写文本",
                segments=[
                    TranscriptSegment(start=0.0, end=2.0, text=f"这是视频 {i}"),
                    TranscriptSegment(start=2.0, end=4.0, text="的转写文本")
                ],
                raw={}
            )
            
            # 写入缓存
            success = clean_redis.set(
                cache_key,
                json.dumps(asdict(transcript), ensure_ascii=False),
                ttl=604800
            )
            assert success is True
        
        # 验证所有缓存都写入成功
        for i in range(3):
            cache_key = f"test:transcript:playlist_video{i}"
            cached_data = clean_redis.get(cache_key)
            assert cached_data is not None
            
            data = json.loads(cached_data)
            assert f"视频 {i}" in data["full_text"]
    
    def test_playlist_cache_hit_rate(self, clean_redis):
        """测试合集缓存命中率"""
        # 模拟 5 个视频，其中 3 个有缓存
        total_videos = 5
        cached_videos = 3
        
        # 写入前 3 个视频的缓存
        for i in range(cached_videos):
            cache_key = f"test:transcript:playlist_video{i}"
            transcript = TranscriptResult(
                language="zh",
                full_text=f"缓存文本 {i}",
                segments=[],
                raw={}
            )
            clean_redis.set(
                cache_key,
                json.dumps(asdict(transcript), ensure_ascii=False),
                ttl=604800
            )
        
        # 检查缓存命中情况
        cache_hit = 0
        cache_miss = 0
        
        for i in range(total_videos):
            cache_key = f"test:transcript:playlist_video{i}"
            if clean_redis.exists(cache_key):
                cache_hit += 1
            else:
                cache_miss += 1
        
        assert cache_hit == cached_videos
        assert cache_miss == (total_videos - cached_videos)
        
        # 计算命中率
        hit_rate = cache_hit / total_videos
        assert hit_rate == 0.6  # 60%


class TestCachePerformance:
    """缓存性能测试"""
    
    def test_cache_read_performance(self, clean_redis):
        """测试缓存读取性能"""
        cache_key = "test:perf:read"
        value = "x" * 1000  # 1KB
        
        clean_redis.set(cache_key, value)
        
        # 测试 100 次读取
        start_time = time.time()
        for _ in range(100):
            clean_redis.get(cache_key)
        elapsed = time.time() - start_time
        
        # 平均每次读取应该小于 10ms
        avg_time = elapsed / 100
        assert avg_time < 0.01
    
    def test_cache_write_performance(self, clean_redis):
        """测试缓存写入性能"""
        value = "x" * 1000  # 1KB
        
        # 测试 100 次写入
        start_time = time.time()
        for i in range(100):
            cache_key = f"test:perf:write{i}"
            clean_redis.set(cache_key, value)
        elapsed = time.time() - start_time
        
        # 平均每次写入应该小于 10ms
        avg_time = elapsed / 100
        assert avg_time < 0.01
        
        # 清理
        for i in range(100):
            clean_redis.delete(f"test:perf:write{i}")
