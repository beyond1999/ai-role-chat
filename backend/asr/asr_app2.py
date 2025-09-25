import asyncio
from collections import deque
from typing import Optional, Deque, Tuple, List, Any, Dict

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import uvicorn
import json

# =========================
# 配置
# =========================
SAMPLE_RATE = 16000                 # 与前端一致（PCM Int16 16kHz 单声道）
RING_SECONDS = 15                   # 环形缓冲区长度（秒）
TICK_SECONDS = 1.0                  # 解码频率（秒）：每秒跑一次
MODEL_PATH = "../../models/faster-whisper-small"   # 可切 medium/large-v2
WHISPER_DEVICE = "cuda"             # "cuda" / "cpu"
WHISPER_COMPUTE_TYPE = "int8"       # "float16"/"int8"/"int8_float16" 等
USE_VAD = True                      # faster-whisper 自带 VAD，建议开
BEAM_SIZE = 1                       # 1=贪心解码，>1=beam search（更准更慢）
LANGUAGE = None                     # 设为 "zh"/"en" 可锁定语言；None=自动
# 可选：只对最近 N 秒做解码（降低延迟避免重复）
DECODE_WINDOW_SECONDS = 8

# =========================
# 模型初始化（进程级别只加载一次）
# =========================
model = WhisperModel(
    MODEL_PATH,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)

app = FastAPI()

# =========================
# 会话对象：保存音频环形缓冲和上次结果
# =========================
class Session:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.ring: Deque[float] = deque(maxlen=sample_rate * RING_SECONDS)
        self._closed = False
        self.last_partial_text: str = ""      # 上次发给前端的 partial，做去抖
        self.last_final_offset: int = 0       # 记录已“最终化”的样本帧数（可选）

    def add_pcm_i16(self, pcm_i16: bytes):
        # bytes -> int16 -> float32 [-1, 1]
        x = np.frombuffer(pcm_i16, dtype=np.int16).astype(np.float32) / 32768.0
        self.ring.extend(x.tolist())

    def snapshot(self) -> Optional[np.ndarray]:
        """取一个快照，避免边收边解读造成的数据抖动。"""
        if not self.ring:
            return None
        return np.array(self.ring, dtype=np.float32)

    def close(self):
        self._closed = True

# =========================
# WebSocket 路由
# =========================
@app.websocket("/ws")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    sess = Session()

    # ---- 接收端：读取 config + 连续 PCM 帧 ----
    async def receiver():
        try:
            # 1) 可选的 config（JSON）
            msg = await ws.receive()
            if msg.get("type") == "websocket.receive":
                if (text := msg.get("text")):
                    try:
                        cfg = json.loads(text)
                        if isinstance(cfg, dict) and cfg.get("op") == "config":
                            # 允许客户端传 {"op":"config","sampleRate":16000}
                            # 此处我们简单忽略/或校验后记录
                            sr = int(cfg.get("sampleRate", SAMPLE_RATE))
                            if sr != SAMPLE_RATE:
                                # 前后端采样率不一致会出错；你也可以在此重设 Session
                                await ws.send_text(json.dumps({
                                    "type": "warning",
                                    "message": f"sampleRate {sr} != server {SAMPLE_RATE}, using server rate."
                                }))
                    except Exception:
                        # 收到文本但不是 JSON，就当控制指令用
                        if text == "__stop__":
                            return

            # 2) 连续接收 PCM 二进制
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.receive":
                    if (binary := message.get("bytes")) is not None:
                        sess.add_pcm_i16(binary)
                    elif (text := message.get("text")):
                        if text == "__stop__":
                            break
                        # 其他控制指令预留
                else:
                    break
        except WebSocketDisconnect:
            pass
        finally:
            sess.close()

    # ---- 解码端：每 tick 取环形缓冲的片段跑一次 whisper ----
    async def transcriber():
        try:
            while not sess._closed:
                await asyncio.sleep(TICK_SECONDS)

                audio = sess.snapshot()
                if audio is None or len(audio) < SAMPLE_RATE * 0.8:  # 少于 ~0.8s 不解码，降空跑
                    continue

                # 仅解码最近 N 秒，降低重复与延迟
                if DECODE_WINDOW_SECONDS is not None:
                    max_len = int(SAMPLE_RATE * DECODE_WINDOW_SECONDS)
                    if len(audio) > max_len:
                        audio = audio[-max_len:]

                # 运行 faster-whisper
                segments, info = model.transcribe(
                    audio,
                    language=LANGUAGE,
                    beam_size=BEAM_SIZE,
                    vad_filter=USE_VAD,
                    vad_parameters=dict(min_silence_duration_ms=200),
                    condition_on_previous_text=False,   # 流式更稳
                    initial_prompt=None,
                    word_timestamps=False,
                )

                # 拼装文本（partial）与时间戳（可作为 final 片段）
                seg_texts: List[str] = []
                seg_ts: List[Tuple[float, float, str]] = []
                for seg in segments:  # segments 是个迭代器
                    seg_texts.append(seg.text)
                    seg_ts.append((seg.start, seg.end, seg.text))

                partial_text = "".join(seg_texts).strip()

                # 去抖：重复内容就别发了
                if partial_text and partial_text != sess.last_partial_text:
                    sess.last_partial_text = partial_text
                    await ws.send_text(json.dumps({
                        "type": "partial",
                        "text": partial_text,
                        "avg_logprob": getattr(info, "avg_logprob", None),
                        "language": getattr(info, "language", None),
                    }))

                # 简单“最终化”策略：
                # 当尾部静音（VAD 判空）或文本稳定时，可以把本轮片段作为 final 发出
                # 这里示例：若 partial 非空且长度较长，就发 final（实际中可结合能量/静音判断）
                if len(partial_text) > 8:
                    await ws.send_text(json.dumps({
                        "type": "final",
                        "text": partial_text,
                        "segments": [
                            {"start": s, "end": e, "text": t} for (s, e, t) in seg_ts
                        ],
                    }))
                    # final 后清空 partial 去抖，下一轮重新累计
                    sess.last_partial_text = ""
        except WebSocketDisconnect:
            pass
        finally:
            sess.close()

    # 并发跑“接收端+解码端”
    recv_task = asyncio.create_task(receiver())
    trans_task = asyncio.create_task(transcriber())
    done, pending = await asyncio.wait(
        {recv_task, trans_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()

if __name__ == "__main__":
    # 用命令行起更好：python app.py 或 uvicorn app:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
