# BiliNote Redis 集成方案

## 一、项目现状与问题分析

### 1.1 核心业务流程
```
视频链接 → 下载音频 → 转写文字 → GPT总结 → 生成Markdown笔记
```

### 1.2 当前技术架构
- **数据库**：SQLite（任务记录）
- **缓存**：文件系统（JSON文件）
- **任务状态**：文件系统（`{task_id}.status.json`）
- **部署方式**：单机部署
- **依赖**：已安装 `redis==5.2.1`

### 1.3 核心问题诊断

#### 问题1：任务状态查询性能差 ⭐⭐⭐⭐⭐
**现象**：
- 前端每秒轮询读取 `{task_id}.status.json` 文件
- 文件 I/O 平均耗时 5-10ms
- 磁盘读写频繁

**影响**：
- 用户体验差，状态更新有延迟感
- 服务器磁盘 I/O 负载高

#### 问题2：缓存无过期机制 ⭐⭐⭐⭐⭐
**现象**：
- 音频文件、转写结果、Markdown 文件永久保存在 `note_results/` 目录
- 目录持续膨胀，无自动清理机制

**影响**：
- 磁盘空间浪费
- 需要手动清理，运维成本高

#### 问题3：缓存查询效率低 ⭐⭐⭐⭐
**现象**：
- 相同视频重复处理时，需要遍历文件系统查找缓存
- 缓存键基于 `platform_videoid_quality`，但查询需要文件 I/O

**影响**：
- 缓存命中时仍需 10-50ms 查询时间
- 响应速度慢

---

## 二、Redis 解决方案

### 2.1 设计原则
1. **最小改动**：不破坏现有代码结构
2. **性能优先**：优先解决高频操作
3. **渐进式**：保留文件系统作为降级方案
4. **自动化**：利用 TTL 自动清理过期数据

### 2.2 核心应用场景

#### 场景1：任务状态管理 ⭐⭐⭐⭐⭐

**数据结构**：Hash
```
Key: task:{task_id}
Fields:
  - status: "DOWNLOADING" | "TRANSCRIBING" | "SUCCESS" | "FAILED"
  - message: 错误信息或进度描述
  - updated_at: 更新时间戳
TTL: 24小时
```

**性能提升**：
- 查询时间：5-10ms → 0.5-1ms（提升 10 倍）
- 自动过期清理，无需手动维护

**实现位置**：
- 修改：`NoteGenerator._update_status()`
- 修改：`GET /task_status/{task_id}` 接口

---

#### 场景2：转写结果缓存 ⭐⭐⭐⭐⭐

**数据结构**：String（JSON）
```
Key: transcript:{platform}:{video_id}:{quality}
Value: TranscriptResult 的 JSON 序列化
TTL: 7天
```

**性能提升**：
- 缓存命中时，跳过转写步骤（节省 30-120 秒）
- 查询时间：10-50ms → 1-2ms

**实现位置**：
- 修改：`NoteGenerator._transcribe_audio()`

---

#### 场景3：音频元信息缓存 ⭐⭐⭐⭐

**数据结构**：String（JSON）
```
Key: audio:{platform}:{video_id}:{quality}
Value: AudioDownloadResult 的 JSON 序列化
TTL: 7天
```

**性能提升**：
- 缓存命中时，跳过视频解析和下载（节省 5-30 秒）
- 查询时间：10-50ms → 1-2ms

**实现位置**：
- 修改：`NoteGenerator._download_media()`

---

#### 场景4：Markdown 结果缓存 ⭐⭐⭐

**数据结构**：String
```
Key: markdown:{task_id}
Value: Markdown 文本
TTL: 30天
```

**性能提升**：
- 用户重新查看历史笔记时，无需读取文件

**实现位置**：
- 修改：`NoteGenerator.generate()` 完成后写入
- 修改：`GET /task_status/{task_id}` 接口优先读取

---

### 2.3 不推荐的场景

| 场景 | 原因 |
|------|------|
| Celery 任务队列 | FastAPI BackgroundTasks 已满足需求，无需引入额外复杂度 |
| API 限流 | 单用户场景，API 自带限流已足够 |
| 实时转写会话 | 使用频率低，内存管理即可 |
| 分布式锁 | 单机部署，无并发竞争问题 |

---

## 三、实施方案

### 3.1 环境配置（已完成 ✅）

**Redis 服务器**：
- 地址：192.168.127.128:6379
- 版本：8.0.2
- 状态：运行正常

**项目配置**：
```env
# backend/.env
REDIS_HOST=192.168.127.128
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB_TASK=0        # 任务状态
REDIS_DB_CACHE=1       # 缓存
```

---

### 3.2 代码实现（分阶段）

#### 阶段一：Redis 客户端封装（30分钟）

**文件**：`backend/app/utils/redis_client.py`

**核心功能**：
- 单例模式管理连接
- 支持多 DB 切换
- 提供 `ping()` 检测可用性
- 连接失败时返回 None，自动降级

---

#### 阶段二：任务状态管理（1小时）

**修改文件**：
- `backend/app/services/note.py` → `_update_status()` 方法
- `backend/app/routers/note.py` → `GET /task_status/{task_id}` 接口

**实现逻辑**：
1. 写入状态时：优先写 Redis，失败时降级到文件
2. 读取状态时：优先读 Redis，未命中时读文件
3. 设置 24 小时 TTL

---

#### 阶段三：缓存优化（2小时）

**修改方法**：
- `_transcribe_audio()`：转写结果缓存
- `_download_media()`：音频元信息缓存
- `generate()`：Markdown 结果缓存

**查询顺序**：
```
Redis → 文件 → 执行操作
```

**写入策略**：
```
Redis + 文件（双写保证数据安全）
```

**TTL 设置**：
- 转写结果：7天
- 音频元信息：7天
- Markdown：30天

---

### 3.3 降级策略

**设计原则**：Redis 不可用时，系统仍能正常运行

**实现方式**：
```python
# 初始化时检测
redis_available = RedisClient.ping()

# 使用时判断
if redis_available:
    try:
        # 尝试 Redis 操作
        result = redis.get(key)
    except:
        # 失败时降级到文件
        result = read_from_file()
else:
    # 直接使用文件系统
    result = read_from_file()
```

---

## 四、性能对比

### 4.1 任务状态查询

| 指标 | 文件系统 | Redis | 提升 |
|------|---------|-------|------|
| 平均响应时间 | 5-10ms | 0.5-1ms | 10倍 |
| 自动清理 | ❌ | ✅ 24小时 | - |

### 4.2 缓存查询

| 指标 | 文件系统 | Redis | 提升 |
|------|---------|-------|------|
| 查询时间 | 10-50ms | 1-2ms | 10-20倍 |
| 过期管理 | 手动 | 自动 | - |

### 4.3 整体流程

| 场景 | 优化前 | 优化后 | 节省时间 |
|------|--------|--------|----------|
| 首次处理 | 60-180秒 | 60-180秒 | 0秒 |
| 缓存命中（转写） | 60-180秒 | 30-60秒 | 30-120秒 |
| 缓存命中（全部） | 60-180秒 | 1-2秒 | 58-178秒 |

---

## 五、监控与维护

### 5.1 监控接口

**接口**：`GET /api/redis/info`

**返回信息**：
- Redis 连接状态
- 已用内存
- 连接客户端数
- 总命令数

### 5.2 缓存清理

**自动清理**（TTL）：
- 任务状态：24小时
- 转写结果：7天
- 音频元信息：7天
- Markdown：30天

**手动清理**：
```bash
# 清理所有任务状态
redis-cli -h 192.168.127.128 --scan --pattern "task:*" | xargs redis-cli -h 192.168.127.128 del

# 清理所有缓存
redis-cli -h 192.168.127.128 FLUSHDB

# 查看内存使用
redis-cli -h 192.168.127.128 INFO memory
```

### 5.3 故障处理

**Redis 连接失败**：
- 自动降级到文件系统
- 记录警告日志
- 不影响核心功能

**Redis 内存不足**：
- 配置 `maxmemory-policy allkeys-lru`
- 自动淘汰最少使用的键

---

## 六、部署配置

### 6.1 开发环境
- 使用远程 Redis（192.168.127.128:6379）
- 开启降级策略

### 6.2 生产环境（Docker Compose）

```yaml
services:
  redis:
    image: redis:7-alpine
    container_name: bilinote-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: >
      redis-server
      --appendonly yes
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
    restart: unless-stopped
  
  backend:
    build: ./backend
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis

volumes:
  redis-data:
```

**配置说明**：
- `appendonly yes`：开启 AOF 持久化
- `maxmemory 512mb`：限制最大内存
- `maxmemory-policy allkeys-lru`：内存不足时 LRU 淘汰

---

## 七、实施时间表

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 第1天 | Redis 客户端封装 + 任务状态管理 | 1.5小时 |
| 第2天 | 缓存优化（转写、音频、Markdown） | 2小时 |
| 第3天 | 测试 + 监控接口 + 文档 | 1小时 |
| **总计** | | **4.5小时** |

---

## 八、预期收益

### 8.1 性能提升
- ✅ 任务状态查询速度提升 **10倍**（5-10ms → 0.5-1ms）
- ✅ 缓存命中时节省 **30-120秒**
- ✅ 缓存查询速度提升 **10-20倍**

### 8.2 运维优化
- ✅ 自动过期清理，减少 **磁盘占用**
- ✅ 无需手动清理缓存文件
- ✅ 降级策略保证 **系统稳定性**

### 8.3 用户体验
- ✅ 状态更新更及时
- ✅ 重复视频处理更快
- ✅ 系统响应更流畅

---

## 九、总结

### 9.1 核心价值
通过引入 Redis，解决 BiliNote 的三大痛点：
1. **任务状态查询慢** → 查询速度提升 10 倍
2. **缓存无过期机制** → TTL 自动清理，减少磁盘占用
3. **缓存查询效率低** → 内存查询，速度提升 10-20 倍

### 9.2 实施建议
- **优先级**：任务状态管理 > 转写缓存 > 音频缓存 > Markdown 缓存
- **策略**：渐进式集成，保留降级方案
- **时间**：预计 **4.5 小时**完成

### 9.3 风险控制
- Redis 不可用时自动降级到文件系统
- 双写策略保证数据安全
- 配置 LRU 淘汰策略防止内存溢出
