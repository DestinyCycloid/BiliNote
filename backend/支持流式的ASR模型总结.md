# 支持流式输出的 ASR 模型总结

经过深入检索，以下是支持流式转写的 ASR 模型和方案：

## 1. 开源模型

### 1.1 whisper.cpp（支持流式）

**项目**：https://github.com/ggml-org/whisper.cpp

**特点**：
- ✅ 支持流式转写（有 `examples/stream` 示例）
- ✅ C/C++ 实现，性能优秀
- ✅ 支持多平台（iOS、Android、Web等）
- ✅ 支持 VAD（语音活动检测）
- ✅ 可以实时从麦克风输入

**流式示例**：
```bash
# 编译流式示例
make stream

# 运行流式转写（从麦克风）
./stream -m models/ggml-base.en.bin -t 8 --step 500 --length 5000
```

**Python 绑定**：
```python
# 使用 pywhispercpp
from pywhispercpp.model import Model

model = Model('base.en', n_threads=6)

# 流式转写
for segment in model.transcribe_stream():
    print(segment.text)
```

### 1.2 Faster-Whisper（部分支持流式）

**项目**：https://github.com/SYSTRAN/faster-whisper

**特点**：
- ✅ 基于 CTranslate2，速度快
- ⚠️ 不是真正的流式，但可以分段处理
- ✅ 支持 VAD 分段

**使用方式**：
```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda")

# 使用 VAD 分段处理（模拟流式）
segments, info = model.transcribe(
    "audio.mp3",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500)
)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### 1.3 NVIDIA Parakeet（支持流式）

**模型**：nvidia/parakeet-tdt-0.6b-v3

**特点**：
- ✅ 专为流式设计（TDT = Transducer）
- ✅ 低延迟
- ✅ 支持实时转写
- ✅ 600M 参数，轻量级

**使用方式**：
```python
from nemo.collections.asr.models import EncDecRNNTBPEModel

model = EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

# 流式转写
for chunk in audio_chunks:
    transcription = model.transcribe([chunk], batch_size=1)
    print(transcription[0])
```

### 1.4 FunASR Paraformer-streaming（支持流式）

**项目**：https://github.com/alibaba-damo-academy/FunASR

**特点**：
- ✅ 阿里达摩院开源
- ✅ 专门的流式版本
- ✅ 支持中文
- ✅ 低延迟

**使用方式**：
```python
from funasr import AutoModel

# 使用流式版本的 Paraformer
model = AutoModel(
    model="paraformer-zh-streaming",
    device="cuda"
)

# 流式转写
cache = {}
for audio_chunk in audio_chunks:
    res = model.generate(
        input=[audio_chunk],
        cache=cache,  # 保持上下文
        is_final=False
    )
    print(res[0]["text"])

# 最后一个块
res = model.generate(
    input=[last_chunk],
    cache=cache,
    is_final=True
)
```

### 1.5 Voxtral-Mini-4B-Realtime（支持实时）

**模型**：mistralai/Voxtral-Mini-4B-Realtime-2602

**特点**：
- ✅ Mistral AI 最新发布（2026年2月）
- ✅ 专为实时设计
- ✅ 4B 参数
- ✅ 支持多语言

---

## 2. 商业 API（支持流式）

### 2.1 Deepgram（推荐）

**网站**：https://deepgram.com

**特点**：
- ✅ WebSocket 流式 API
- ✅ 低延迟（< 300ms）
- ✅ 支持多语言
- ✅ 免费额度：$200

**使用方式**：
```python
from deepgram import DeepgramClient, LiveTranscriptionEvents

dg_client = DeepgramClient(api_key)
dg_connection = dg_client.listen.websocket.v("1")

def on_message(self, result, **kwargs):
    sentence = result.channel.alternatives[0].transcript
    if len(sentence) > 0:
        print(f"Transcript: {sentence}")

dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
dg_connection.start()

# 发送音频块
for chunk in audio_chunks:
    dg_connection.send(chunk)

dg_connection.finish()
```

### 2.2 阿里云语音服务

**网站**：https://www.aliyun.com/product/nls

**特点**：
- ✅ 实时语音识别
- ✅ WebSocket 流式
- ✅ 支持中文方言
- ✅ 有免费额度

### 2.3 腾讯云语音识别

**网站**：https://cloud.tencent.com/product/asr

**特点**：
- ✅ 实时语音识别
- ✅ 支持中文
- ✅ 有免费额度

### 2.4 讯飞开放平台

**网站**：https://www.xfyun.cn

**特点**：
- ✅ 实时语音转写
- ✅ 中文识别准确率高
- ✅ 有免费额度

---

## 3. Fun-ASR-Nano 的真实情况

### 结论

经过深入检索和测试，**Fun-ASR-Nano 不支持流式传输**。

### 证据

1. **官方 demo 只支持文件路径**
   ```python
   # 只能这样
   res = model.generate(input=["audio.mp3"])
   
   # 不能这样
   res = model.generate(input=[audio_array])  # 报错
   ```

2. **没有流式示例**
   - 官方 GitHub 没有流式示例
   - 只有批量处理的 demo

3. **"real-time" ≠ "streaming"**
   - "real-time transcription" = 快速处理（RTF < 1）
   - "streaming" = 边接收边处理

### Fun-ASR-Nano vs FunASR Paraformer-streaming

| 特性 | Fun-ASR-Nano | Paraformer-streaming |
|------|--------------|---------------------|
| 项目 | Fun-ASR（新模型） | FunASR（工具包） |
| 流式支持 | ❌ 否 | ✅ 是 |
| 输入方式 | 文件路径 | 音频数组 |
| 处理方式 | 批量 | 流式 |
| RTF | 0.436 | < 1 |
| 适用场景 | 离线转写 | 实时字幕 |

---

## 4. 推荐方案

### 4.1 如果需要开源流式方案

**推荐顺序**：
1. **whisper.cpp**（最成熟，跨平台）
2. **FunASR Paraformer-streaming**（中文友好）
3. **NVIDIA Parakeet**（性能好）

### 4.2 如果需要商业流式方案

**推荐顺序**：
1. **Deepgram**（延迟最低，API 最好）
2. **阿里云语音服务**（中文准确率高）
3. **讯飞开放平台**（中文专业）

### 4.3 对于 BiliNote 项目

**当前场景（离线转写）**：
- ✅ 继续使用 Fun-ASR-Nano
- ✅ 性能已经很好（RTF 0.436）
- ✅ 不需要流式

**如果未来需要实时字幕**：
- 方案 1：使用 **whisper.cpp** 的流式功能
- 方案 2：使用 **Deepgram** 的流式 API
- 方案 3：使用 **FunASR Paraformer-streaming**

---

## 5. 流式转写实现示例

### 5.1 whisper.cpp 流式（推荐）

```python
# 安装 pywhispercpp
pip install pywhispercpp

# 使用流式
from pywhispercpp.model import Model

model = Model('base.en', n_threads=6)

# 从麦克风流式转写
for segment in model.transcribe_stream():
    print(f"[{segment.start:.2f}s] {segment.text}")
```

### 5.2 Deepgram 流式（商业）

```python
from deepgram import DeepgramClient, LiveTranscriptionEvents

client = DeepgramClient(api_key="YOUR_API_KEY")
connection = client.listen.websocket.v("1")

def on_message(self, result, **kwargs):
    transcript = result.channel.alternatives[0].transcript
    if transcript:
        print(f"Transcript: {transcript}")

connection.on(LiveTranscriptionEvents.Transcript, on_message)
connection.start()

# 发送音频
with open("audio.wav", "rb") as audio:
    while True:
        chunk = audio.read(8000)
        if not chunk:
            break
        connection.send(chunk)

connection.finish()
```

### 5.3 FunASR Paraformer-streaming（开源）

```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh-streaming",
    device="cuda"
)

cache = {}
for i, chunk in enumerate(audio_chunks):
    is_final = (i == len(audio_chunks) - 1)
    
    res = model.generate(
        input=[chunk],
        cache=cache,
        is_final=is_final
    )
    
    if res and len(res) > 0:
        print(f"Chunk {i}: {res[0]['text']}")
```

---

## 6. 总结

### Fun-ASR-Nano
- ❌ 不支持流式传输
- ✅ 支持快速批量处理
- ✅ 适合离线场景
- ✅ 性能优秀（RTF 0.436）

### 支持流式的开源方案
1. **whisper.cpp** - 最成熟
2. **FunASR Paraformer-streaming** - 中文友好
3. **NVIDIA Parakeet** - 性能好
4. **Voxtral-Mini-4B-Realtime** - 最新

### 支持流式的商业方案
1. **Deepgram** - 最推荐
2. **阿里云语音服务** - 中文准确
3. **腾讯云/讯飞** - 国内选择

---

## 参考资料

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [Deepgram](https://deepgram.com)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [Hugging Face ASR Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition)
