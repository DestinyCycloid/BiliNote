# Redis 配置说明

## 配置文件位置

所有 Redis 配置都在 `backend/.env` 文件中，**没有任何硬编码**。

## 配置项说明

```env
# Redis 服务器地址
REDIS_HOST=192.168.127.128

# Redis 服务器端口
REDIS_PORT=6379

# Redis 密码（如果没有密码，留空即可）
REDIS_PASSWORD=

# Redis 数据库编号配置
REDIS_DB_TASK=0        # 任务状态管理
REDIS_DB_CACHE=1       # 缓存数据
REDIS_DB_QUEUE=2       # 任务队列（预留）
```

## 更换 Redis 服务器

如果需要更换 Redis 服务器，只需修改 `.env` 文件中的配置即可，**无需修改任何代码**。

### 示例 1：切换到本地 Redis

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
```

### 示例 2：切换到远程 Redis（带密码）

```env
REDIS_HOST=redis.example.com
REDIS_PORT=6379
REDIS_PASSWORD=your_password_here
```

### 示例 3：使用不同的 DB 编号

```env
REDIS_DB_TASK=3        # 使用 DB3 存储任务状态
REDIS_DB_CACHE=4       # 使用 DB4 存储缓存
REDIS_DB_QUEUE=5       # 使用 DB5 存储队列
```

## 代码中的使用方式

项目中使用了三种便捷方法来创建 Redis 管理器，自动从环境变量读取配置：

### 1. 任务状态管理

```python
from app.utils.redis_client import RedisManager

# 自动使用 REDIS_DB_TASK 配置的 DB
redis_manager = RedisManager.for_task()
```

### 2. 缓存管理

```python
from app.utils.redis_client import RedisManager

# 自动使用 REDIS_DB_CACHE 配置的 DB
redis_manager = RedisManager.for_cache()
```

### 3. 队列管理（预留）

```python
from app.utils.redis_client import RedisManager

# 自动使用 REDIS_DB_QUEUE 配置的 DB
redis_manager = RedisManager.for_queue()
```

### 4. 手动指定 DB（不推荐）

```python
from app.utils.redis_client import RedisManager

# 手动指定 DB 编号
redis_manager = RedisManager(db=5)
```

## Redis 使用场景

### 1. 任务状态管理（DB0）

- 存储任务的实时状态
- 支持高频率查询（100+ QPS）
- TTL：24 小时自动过期

**使用位置**：
- `backend/app/services/note.py` - `_update_status()` 方法
- `backend/app/routers/note.py` - `GET /task_status/{task_id}` 接口

### 2. 缓存管理（DB1）

- 转写结果缓存（TTL: 7天）
- 音频元信息缓存（TTL: 7天）
- Markdown 结果缓存（TTL: 30天）

**使用位置**：
- `backend/app/services/note.py` - `_download_media()`, `_transcribe_audio()`, `generate()` 方法
- `backend/app/services/playlist_processor.py` - `_transcribe_with_cache_and_retry()` 方法

### 3. 合集处理缓存

合集处理时，每个视频都会检查 Redis 转写缓存：

```python
# 缓存键格式
cache_key = f"transcript:playlist_{video_id}"

# 缓存命中时，跳过转写步骤，节省 30-120 秒
```

**日志输出示例**：
```
[1/10] ✅ 从 Redis 读取转写缓存
[2/10] ✅ 从 Redis 读取转写缓存
[3/10] 开始转写
[3/10] ✅ 转写结果已缓存到 Redis (TTL: 7天)
```

## 验证配置

### 方法 1：运行测试

```bash
cd backend
python -m pytest tests/ -v
```

如果所有测试通过，说明 Redis 配置正确。

### 方法 2：手动验证

```python
from app.utils.redis_client import RedisManager

# 测试任务状态 DB
task_manager = RedisManager.for_task()
print(f"任务状态 DB 可用: {task_manager.available}")
print(f"使用的 DB: {task_manager.db}")

# 测试缓存 DB
cache_manager = RedisManager.for_cache()
print(f"缓存 DB 可用: {cache_manager.available}")
print(f"使用的 DB: {cache_manager.db}")
```

## 故障排查

### 问题 1：连接失败

**错误信息**：
```
❌ Redis 连接失败: [Errno 111] Connection refused
```

**解决方法**：
1. 检查 Redis 服务是否运行
2. 检查 `REDIS_HOST` 和 `REDIS_PORT` 配置
3. 检查防火墙设置
4. 检查网络连接

### 问题 2：认证失败

**错误信息**：
```
❌ Redis 连接失败: NOAUTH Authentication required
```

**解决方法**：
在 `.env` 中设置正确的密码：
```env
REDIS_PASSWORD=your_password
```

### 问题 3：DB 不存在

**错误信息**：
```
❌ Redis 连接失败: invalid DB index
```

**解决方法**：
Redis 默认有 16 个数据库（DB0-DB15），确保配置的 DB 编号在有效范围内：
```env
REDIS_DB_TASK=0        # 有效范围: 0-15
REDIS_DB_CACHE=1       # 有效范围: 0-15
```

## 性能优化建议

### 1. 使用本地 Redis

如果可能，将 Redis 部署在应用服务器本地，减少网络延迟：

```env
REDIS_HOST=localhost
```

### 2. 调整连接超时

如果网络不稳定，可以在代码中调整超时时间（需要修改 `redis_client.py`）：

```python
client = redis.Redis(
    socket_connect_timeout=10,  # 连接超时：10秒
    socket_timeout=10,          # 操作超时：10秒
)
```

### 3. 监控 Redis 性能

使用 Redis 监控接口查看性能指标：

```bash
curl http://localhost:8000/redis/info
```

## 总结

✅ **所有 Redis 配置都在 `.env` 文件中**  
✅ **代码中没有任何硬编码**  
✅ **更换 Redis 只需修改 `.env`**  
✅ **支持密码认证**  
✅ **支持多 DB 配置**  
✅ **自动降级到文件系统**  
✅ **合集处理已集成 Redis 缓存**  

修改配置后，重启应用即可生效，无需修改任何代码。
