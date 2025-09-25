

## 1. 最小可跑通 Demo（Python 本地）

### 🔹 思路

* 用 `sounddevice` 或 `pyaudio` 持续采集麦克风音频 → 得到 **小块 PCM buffer**（比如 0.5 秒）。
* 把 buffer 放进 **ring buffer**（避免频繁 IO）。
* 定时（比如每 1 秒）取出最近 N 秒数据拼成一段，丢给 `WhisperModel.transcribe()`。
* 把识别结果逐步拼接显示。

### 🔹 Demo 代码骨架

```python
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue, threading

# 初始化模型（先 warm-up）
model = WhisperModel("small", device="cuda", compute_type="int8_float16")
print("model loaded")

# 音频参数
SAMPLE_RATE = 16000
BLOCK_SIZE = int(SAMPLE_RATE * 0.5)  # 每次取 0.5 秒

# 环形缓冲队列
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

def transcribe_loop():
    buffer = []
    while True:
        data = q.get()
        buffer.append(data)
        if len(buffer) >= 4:  # 约 2 秒拼一次
            chunk = np.concatenate(buffer, axis=0).flatten()
            buffer.clear()
            # 直接送 numpy array（WhisperModel 支持内存音频）
            segments, _ = model.transcribe(chunk, beam_size=1, vad_filter=False)
            text = "".join(seg.text for seg in segments)
            print("实时转写:", text)

# 启动转写线程
threading.Thread(target=transcribe_loop, daemon=True).start()

# 启动麦克风采集
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback):
    print("🎤 说话吧（Ctrl+C 退出）")
    threading.Event().wait()  # 阻塞主线程
```

👉 这样就能做到：你边说话，程序每隔 2 秒左右就输出一段文字。

---

## 2. 工业级常见做法

### 🔹 架构

* **前端（浏览器 / App）**：录音 API 获取音频流 → 用 WebSocket/gRPC 发送给后端。
* **后端（ASR 服务）**：

  * 使用 ring buffer 累积音频块；
  * 每隔固定窗口调用一次推理；
  * 输出部分结果给前端 → 前端增量更新字幕。

### 🔹 工程要点

* **延迟 vs 准确率**：窗口太短 → 有延迟感小，但上下文不足可能识别差；窗口太长 → 识别稳但延迟高。常用 **1–2 秒窗口，0.5 秒步长**（overlap sliding）。
* **VAD（语音活动检测）**：能避免把静音/噪音送进模型，提升速度和精度。
* **结果拼接**：需要去重和合并，避免前后片段重叠时出现“重复文字”。

### 🔹 框架工具

* WebSocket + FastAPI（轻量级 demo）。
* gRPC streaming（大规模实时系统）。
* 商业系统里常常会配合 **Kafka/Redis** 来做多路音频流分发。

