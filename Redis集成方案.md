# BiliNote Redis 集成方案

## 一、项目现状分析

### 当前架构
- **数据库**：SQLite（存储模型、提供商、视频任务记录）
- **缓存**：文件系统（JSON 文件）
- **任务状态**：文件系统（`{task_id}.status.json`）
- **依赖**：`requirements.txt` 中已包含 `redis==5.2.1`，但未使用

### 存在的问题
1. **文件 I/O 性能瓶颈**：任务状态频繁读写文件
2. **缓存管理困难**：无法自动过期，需手动清理
3. **并发问题**：多进程部署时文件竞争
4. **实时性差**：轮询查询效率低，无法推送更新
5. **分布式支持弱**：无法跨实例共享状态

---

## 二、Redis 应用场景

### 1. 任务状态管理 ⭐⭐⭐⭐⭐

**优先级：最高**

#### 当前实现
```python
# backend/app/services/note.py
def _update_status(self, task_id, status, message=None):
    status_file = NOTE_OUTPUT_DIR / f"{task_id}.status.json"
    data = {"status": status.value, "message": message}
    with status_file.open('w') as f:
        json.dump(data, f)
```

#### Redis 方案
```python
import redis
from datetime import timedelta

class RedisTaskManager:
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True
        )
    
    def update_status(self, task_id: str, status: str, message: str = None, progress: int = None):
        """更新任务状态"""
        key = f"task:{task_id}"
        data = {
            "status": status,
            "updated_at": time.time()
        }
        if message:
            data["message"] = message
        if progress is not None:
            data["progress"] = progress
        
        # 使用 Hash 存储
        self.redis.hset(key, mapping=data)
        
        # 设置过期时间（24小时）
        self.redis.expire(key, 86400)
    
    def get_status(self, task_id: str) -> dict:
        """获取任务状态"""
        key = f"task:{task_id}"
        data = self.redis.hgetall(key)
        if not data:
            return None
        return data
    
    def publish_status(self, task_id: str, status: str):
        """发布状态更新（用于实时推送）"""
        channel = f"task_updates:{task_id}"
        self.redis.publish(channel, json.dumps({
            "task_id": task_id,
            "status": status,
            "timestamp": time.time()
        }))
```

#### 优势
- ✅ 毫秒级读写性能
- ✅ 支持 Pub/Sub 实时推送
- ✅ 自动过期清理
- ✅ 多进程安全

---

### 2. 转写结果缓存 ⭐⭐⭐⭐⭐

**优先级：最高**

#### 当前实现
```python
# 文件缓存
cache_key = f"{platform}_{video_id}_{quality}"
transcript_cache_file = NOTE_OUTPUT_DIR / f"{cache_key}_transcript.json"

if transcript_cache_file.exists():
    data = json.loads(transcript_cache_file.read_text())
    return TranscriptResult(**data)
```

#### Redis 方案
```python
class RedisCacheManager:
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=1,  # 使用独立的 DB
            decode_responses=True
        )
    
    def cache_transcript(self, platform: str, video_id: str, quality: str, transcript: TranscriptResult):
        """缓存转写结果（7天过期）"""
        key = f"transcript:{platform}:{video_id}:{quality}"
        data = json.dumps(asdict(transcript), ensure_ascii=False)
        self.redis.setex(key, 604800, data)  # 7天
    
    def get_transcript(self, platform: str, video_id: str, quality: str) -> TranscriptResult:
        """获取缓存的转写结果"""
        key = f"transcript:{platform}:{video_id}:{quality}"
        data = self.redis.get(key)
        if data:
            return TranscriptResult(**json.loads(data))
        return None
    
    def cache_audio_meta(self, platform: str, video_id: str, quality: str, audio_meta: AudioDownloadResult):
        """缓存音频元信息（7天过期）"""
        key = f"audio:{platform}:{video_id}:{quality}"
        data = json.dumps(asdict(audio_meta), ensure_ascii=False)
        self.redis.setex(key, 604800, data)
    
    def cache_markdown(self, task_id: str, markdown: str):
        """缓存 Markdown 结果（30天过期）"""
        key = f"markdown:{task_id}"
        self.redis.setex(key, 2592000, markdown)
```

#### 优势
- ✅ 查询速度提升 100+ 倍
- ✅ 自动过期管理
- ✅ 支持 LRU 淘汰策略
- ✅ 内存占用可控

---

### 3. 合集处理任务队列 ⭐⭐⭐⭐⭐

**优先级：高**

#### 当前实现
```python
# 使用线程池并行处理
processor = SimpleThreadPipeline(transcriber=self.transcriber, max_workers=None)
markdowns = processor.process_playlist(audio_results, gpt, task_id)
```

#### Redis 方案（使用 Celery）
```python
# backend/app/tasks/celery_app.py
from celery import Celery

celery_app = Celery(
    'bilinote',
    broker=f'redis://{os.getenv("REDIS_HOST", "localhost")}:6379/2',
    backend=f'redis://{os.getenv("REDIS_HOST", "localhost")}:6379/3'
)

@celery_app.task(bind=True, max_retries=3)
def transcribe_video_task(self, video_url: str, quality: str, task_id: str):
    """异步转写任务"""
    try:
        # 更新进度
        self.update_state(state='PROGRESS', meta={'progress': 0})
        
        # 下载音频
        audio = download_audio(video_url, quality)
        self.update_state(state='PROGRESS', meta={'progress': 30})
        
        # 转写
        transcript = transcribe_audio(audio.file_path)
        self.update_state(state='PROGRESS', meta={'progress': 70})
        
        # GPT 总结
        markdown = summarize_text(transcript)
        self.update_state(state='PROGRESS', meta={'progress': 100})
        
        return {"markdown": markdown, "status": "success"}
    except Exception as exc:
        self.retry(exc=exc, countdown=60)

@celery_app.task
def process_playlist_task(video_urls: list, task_id: str):
    """合集处理任务"""
    # 创建子任务组
    job = group(
        transcribe_video_task.s(url, "medium", f"{task_id}_{i}")
        for i, url in enumerate(video_urls)
    )
    result = job.apply_async()
    return result.get()
```

#### 使用方式
```python
# 提交任务
task = transcribe_video_task.delay(video_url, quality, task_id)

# 查询进度
result = celery_app.AsyncResult(task.id)
print(result.state, result.info)
```

#### 优势
- ✅ 分布式处理（多机器）
- ✅ 自动重试机制
- ✅ 任务优先级控制
- ✅ 实时进度跟踪
- ✅ 失败任务可恢复

---

### 4. API 限流 ⭐⭐⭐⭐

**优先级：中**

#### 场景
防止 Deepgram、Groq 等云端 API 超出配额

#### Redis 方案
```python
class RateLimiter:
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=4,
            decode_responses=True
        )
    
    def check_rate_limit(self, api_name: str, user_id: str, max_requests: int, window: int) -> bool:
        """
        滑动窗口限流
        
        :param api_name: API 名称（如 "deepgram", "groq"）
        :param user_id: 用户 ID
        :param max_requests: 窗口内最大请求数
        :param window: 时间窗口（秒）
        :return: True 表示允许请求，False 表示超限
        """
        key = f"ratelimit:{api_name}:{user_id}"
        current = self.redis.incr(key)
        
        if current == 1:
            self.redis.expire(key, window)
        
        return current <= max_requests
    
    def get_remaining(self, api_name: str, user_id: str, max_requests: int) -> int:
        """获取剩余配额"""
        key = f"ratelimit:{api_name}:{user_id}"
        current = int(self.redis.get(key) or 0)
        return max(0, max_requests - current)

# 使用示例
limiter = RateLimiter()

def call_deepgram_api(user_id: str):
    # Deepgram 限制：每分钟 20 次
    if not limiter.check_rate_limit("deepgram", user_id, 20, 60):
        raise HTTPException(429, "API 调用超限，请稍后重试")
    
    # 调用 API
    result = deepgram.transcribe(...)
    return result
```

---

### 5. 实时转写会话管理 ⭐⭐⭐⭐

**优先级：中**

#### 场景
Deepgram、Paraformer 实时转写的 WebSocket 会话管理

#### Redis 方案
```python
class RealtimeSessionManager:
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=5,
            decode_responses=True
        )
    
    def create_session(self, session_id: str, user_id: str, model: str):
        """创建实时转写会话"""
        key = f"session:{session_id}"
        self.redis.hset(key, mapping={
            "user_id": user_id,
            "model": model,
            "start_time": time.time(),
            "status": "active"
        })
        self.redis.expire(key, 3600)  # 1小时过期
    
    def append_text(self, session_id: str, text: str):
        """追加转写文本"""
        key = f"session:{session_id}:text"
        self.redis.append(key, text + "\n")
        self.redis.expire(key, 3600)
    
    def get_session_text(self, session_id: str) -> str:
        """获取会话的完整转写文本"""
        key = f"session:{session_id}:text"
        return self.redis.get(key) or ""
    
    def close_session(self, session_id: str):
        """关闭会话"""
        key = f"session:{session_id}"
        self.redis.hset(key, "status", "closed")
        self.redis.hset(key, "end_time", time.time())
```

---

### 6. 视频下载去重 ⭐⭐⭐

**优先级：低**

#### 场景
多用户同时下载同一视频时，避免重复下载

#### Redis 方案
```python
class DownloadLockManager:
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=6,
            decode_responses=True
        )
    
    def acquire_lock(self, video_id: str, timeout: int = 300) -> bool:
        """获取下载锁"""
        key = f"lock:download:{video_id}"
        return self.redis.set(key, 1, nx=True, ex=timeout)
    
    def release_lock(self, video_id: str):
        """释放下载锁"""
        key = f"lock:download:{video_id}"
        self.redis.delete(key)
    
    def wait_for_download(self, video_id: str, max_wait: int = 300):
        """等待其他进程完成下载"""
        key = f"lock:download:{video_id}"
        start = time.time()
        while time.time() - start < max_wait:
            if not self.redis.exists(key):
                return True
            time.sleep(1)
        return False

# 使用示例
lock_manager = DownloadLockManager()

def download_video(video_id: str):
    if lock_manager.acquire_lock(video_id):
        try:
            # 执行下载
            result = downloader.download(video_url)
            return result
        finally:
            lock_manager.release_lock(video_id)
    else:
        # 等待其他进程完成
        if lock_manager.wait_for_download(video_id):
            # 从缓存读取
            return get_cached_video(video_id)
        else:
            raise TimeoutError("下载超时")
```

---

## 三、实施方案

### 阶段一：基础集成（1-2天）

#### 1. 安装和配置
```bash
# 安装 Redis（已在 requirements.txt 中）
pip install redis==5.2.1

# 启动 Redis 服务
# Windows: 下载 Redis for Windows
# Linux/Mac: sudo systemctl start redis
```

#### 2. 配置文件
```python
# backend/.env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB_TASK=0        # 任务状态
REDIS_DB_CACHE=1       # 缓存
REDIS_DB_QUEUE=2       # 任务队列
```

#### 3. 创建 Redis 客户端
```python
# backend/app/utils/redis_client.py
import redis
import os
from typing import Optional

class RedisClient:
    _instance: Optional[redis.Redis] = None
    
    @classmethod
    def get_instance(cls, db: int = 0) -> redis.Redis:
        """获取 Redis 单例"""
        if cls._instance is None:
            cls._instance = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                password=os.getenv("REDIS_PASSWORD", None),
                db=db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
        return cls._instance
    
    @classmethod
    def ping(cls) -> bool:
        """检查 Redis 连接"""
        try:
            return cls.get_instance().ping()
        except:
            return False
```

#### 4. 修改任务状态管理
```python
# backend/app/services/note.py
from app.utils.redis_client import RedisClient

class NoteGenerator:
    def __init__(self):
        # ... 原有代码
        self.redis = RedisClient.get_instance(db=0)
        self.use_redis = RedisClient.ping()
    
    def _update_status(self, task_id: str, status: TaskStatus, message: str = None):
        """更新任务状态（优先使用 Redis）"""
        if not task_id:
            return
        
        if self.use_redis:
            # 使用 Redis
            key = f"task:{task_id}"
            data = {
                "status": status.value,
                "updated_at": str(time.time())
            }
            if message:
                data["message"] = message
            
            self.redis.hset(key, mapping=data)
            self.redis.expire(key, 86400)
        else:
            # 降级到文件系统
            status_file = NOTE_OUTPUT_DIR / f"{task_id}.status.json"
            # ... 原有文件写入逻辑
```

---

### 阶段二：缓存优化（2-3天）

#### 1. 实现缓存管理器
```python
# backend/app/utils/cache_manager.py
from app.utils.redis_client import RedisClient
import json
from typing import Optional, Any

class CacheManager:
    def __init__(self):
        self.redis = RedisClient.get_instance(db=1)
        self.use_redis = RedisClient.ping()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if self.use_redis:
            data = self.redis.get(key)
            return json.loads(data) if data else None
        return None
    
    def set(self, key: str, value: Any, ttl: int = 604800):
        """设置缓存（默认7天）"""
        if self.use_redis:
            self.redis.setex(key, ttl, json.dumps(value, ensure_ascii=False))
    
    def delete(self, key: str):
        """删除缓存"""
        if self.use_redis:
            self.redis.delete(key)
```

#### 2. 修改转写缓存逻辑
```python
# backend/app/services/note.py
from app.utils.cache_manager import CacheManager

class NoteGenerator:
    def __init__(self):
        # ... 原有代码
        self.cache = CacheManager()
    
    def _transcribe_audio(self, audio_file: str, transcript_cache_file: Path, status_phase: TaskStatus):
        """转写音频（优先使用 Redis 缓存）"""
        task_id = transcript_cache_file.stem.split("_")[0]
        self._update_status(task_id, status_phase)
        
        # 尝试从 Redis 缓存读取
        cache_key = f"transcript:{audio_file}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.info(f"从 Redis 缓存读取转写结果")
            return TranscriptResult(**cached)
        
        # 尝试从文件缓存读取
        if transcript_cache_file.exists():
            # ... 原有文件读取逻辑
        
        # 执行转写
        transcript = self.transcriber.transcript(file_path=audio_file)
        
        # 写入 Redis 缓存
        self.cache.set(cache_key, asdict(transcript), ttl=604800)
        
        # 写入文件缓存（降级方案）
        transcript_cache_file.write_text(...)
        
        return transcript
```

---

### 阶段三：任务队列（3-5天，可选）

#### 1. 安装 Celery
```bash
pip install celery==5.3.4
```

#### 2. 创建 Celery 应用
```python
# backend/app/tasks/celery_app.py
from celery import Celery
import os

celery_app = Celery(
    'bilinote',
    broker=f'redis://{os.getenv("REDIS_HOST", "localhost")}:6379/2',
    backend=f'redis://{os.getenv("REDIS_HOST", "localhost")}:6379/3'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1小时超时
    worker_prefetch_multiplier=1,
)
```

#### 3. 启动 Celery Worker
```bash
# 开发环境
celery -A app.tasks.celery_app worker --loglevel=info

# 生产环境
celery -A app.tasks.celery_app worker --loglevel=info --concurrency=4
```

---

## 四、部署配置

### Docker Compose 配置
```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: bilinote-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
  
  backend:
    build: ./backend
    container_name: bilinote-backend
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis
    ports:
      - "8483:8483"
  
  celery-worker:
    build: ./backend
    container_name: bilinote-celery
    command: celery -A app.tasks.celery_app worker --loglevel=info
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis
      - backend

volumes:
  redis-data:
```

---

## 五、性能对比

### 任务状态查询
| 方案 | 平均响应时间 | QPS |
|------|------------|-----|
| 文件系统 | 5-10ms | ~200 |
| Redis | 0.5-1ms | ~10000 |

### 缓存命中
| 方案 | 查询时间 | 内存占用 |
|------|---------|---------|
| 文件系统 | 10-50ms | 磁盘 |
| Redis | 1-2ms | 内存 |

---

## 六、监控和维护

### Redis 监控
```python
# backend/app/routers/system.py
@router.get("/redis/info")
def get_redis_info():
    """获取 Redis 状态"""
    redis_client = RedisClient.get_instance()
    info = redis_client.info()
    return {
        "connected": True,
        "used_memory": info.get("used_memory_human"),
        "connected_clients": info.get("connected_clients"),
        "total_commands": info.get("total_commands_processed"),
        "uptime_days": info.get("uptime_in_days")
    }
```

### 缓存清理
```python
# 清理过期任务状态
redis-cli --scan --pattern "task:*" | xargs redis-cli del

# 清理所有缓存
redis-cli FLUSHDB
```

---

## 七、总结

### 推荐实施顺序
1. **阶段一**：任务状态管理 + 基础缓存（必须）
2. **阶段二**：完整缓存系统（推荐）
3. **阶段三**：Celery 任务队列（可选，适合高并发场景）

### 预期收益
- ✅ 任务状态查询性能提升 **10倍**
- ✅ 缓存命中率提升至 **80%+**
- ✅ 支持分布式部署
- ✅ 自动过期管理，减少磁盘占用
- ✅ 实时状态推送，提升用户体验

### 风险控制
- 保留文件系统作为降级方案
- Redis 不可用时自动切换到文件缓存
- 定期备份 Redis 数据（RDB + AOF）
