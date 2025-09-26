import asyncio
from collections import deque
from typing import Optional, Deque, Tuple, List, Any, Dict

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import uvicorn
import json
from backend.llm.llm import

# =========================
# 配置
# =========================
SAMPLE_RATE = 16000                 # 与前端一致（PCM Int16 16kHz 单声道）
RING_SECONDS = 15                   # 环形缓冲区长度（秒）
# TICK_SECONDS = 1                 # 解码频率（秒）：每秒跑一次
MODEL_PATH = "../../models/faster-whisper-small"   # 可切 medium/large-v2
WHISPER_DEVICE = "cuda"             # "cuda" / "cpu"
WHISPER_COMPUTE_TYPE = "int8"       # "float16"/"int8"/"int8_float16" 等
USE_VAD = True                      # faster-whisper 自带 VAD，建议开
BEAM_SIZE = 1                       # 1=贪心解码，>1=beam search（更准更慢）
LANGUAGE = None                     # 设为 "zh"/"en" 可锁定语言；None=自动
# 可选：只对最近 N 秒做解码（降低延迟避免重复）
DECODE_WINDOW_SECONDS = 8

# ======= 增加：端点器配置（NEW） =======
TICK_SECONDS = 0.25                     # 更细的tick
END_SILENCE_MS = 800                    # 说完后判定结束需要的连续静音时长
SHORT_PAUSE_MS = 300                    # 有结束标点时，较短静音即可final
STABLE_NOCHANGE_MS = 1500               # 文本在这么久没有变化 -> 也final
SILENCE_RMS_THRESH = 0.005              # 静音阈值（RMS），按需微调
SILENCE_WINDOW_MS = 300                 # 静音判定看最近这段音频
END_PUNCTS = set("。.!！？?")
# =========================
# 模型初始化（进程级别只加载一次）
# =========================
model = WhisperModel(
    MODEL_PATH,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)


# ======= 端点器状态机（NEW） =======
class Endpointor:
    def __init__(self):
        self.text_buffer = ""           # 累计当前句子的文本
        self.last_text = ""             # 上一轮 partial
        self.last_change_ts = None      # partial 最近一次变化的「时间」
        self.silence_acc_ms = 0         # 连续静音累计
        self.in_speech = False          # 是否在说话中

    def reset_sentence(self):
        self.text_buffer = ""
        self.last_text = ""
        self.last_change_ts = None
        self.silence_acc_ms = 0
        self.in_speech = False

    def update(
        self,
        now_ms: float,
        partial_text: str,
        is_silence: bool,
        tick_ms: float
    ) -> tuple[bool, str]:
        """
        返回 (should_finalize, final_text)
        """
        # 是否在讲话
        if not is_silence and partial_text:
            self.in_speech = True

        # 记录 partial 是否变化
        if partial_text != self.last_text:
            self.last_text = partial_text
            self.last_change_ts = now_ms

        # 静音累计
        if is_silence:
            self.silence_acc_ms += tick_ms
        else:
            self.silence_acc_ms = 0

        # 规则1：结束标点 + 短暂停
        if partial_text and partial_text[-1] in END_PUNCTS and self.silence_acc_ms >= SHORT_PAUSE_MS:
            final = partial_text.strip()
            self.reset_sentence()
            return True, final

        # 规则2：长静音（人停下来说话）
        if self.in_speech and self.silence_acc_ms >= END_SILENCE_MS:
            final = partial_text.strip()
            self.reset_sentence()
            return True, final

        # 规则3：文本稳定（长时间没变化）
        if self.last_change_ts is not None and (now_ms - self.last_change_ts) >= STABLE_NOCHANGE_MS and partial_text:
            final = partial_text.strip()
            self.reset_sentence()
            return True, final

        return False, ""

app = FastAPI()
# ======= 工具：最近窗口RMS计算（NEW） =======
def rms_recent(audio: np.ndarray, sample_rate: int, window_ms: int) -> float:
    if audio is None or len(audio) == 0:
        return 1.0
    n = int(sample_rate * (window_ms / 1000.0))
    if n <= 0:
        return 1.0
    tail = audio[-n:] if len(audio) >= n else audio
    return float(np.sqrt(np.mean(tail * tail)) + 1e-12)
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
        ep = Endpointor()                                  # NEW
        try:
            now_ms = 0.0
            tick_ms = TICK_SECONDS * 1000.0
            while not sess._closed:
                await asyncio.sleep(TICK_SECONDS)
                now_ms += tick_ms

                audio = sess.snapshot()
                if audio is None or len(audio) < SAMPLE_RATE * 0.3:   # 少于0.3秒就先不跑
                    continue

                # 仅解码最近 N 秒
                if DECODE_WINDOW_SECONDS is not None:
                    max_len = int(SAMPLE_RATE * DECODE_WINDOW_SECONDS)
                    if len(audio) > max_len:
                        audio = audio[-max_len:]

                # 计算是否静音（最近SILENCE_WINDOW_MS窗口）
                rms = rms_recent(audio, SAMPLE_RATE, SILENCE_WINDOW_MS)   # NEW
                is_silence = (rms < SILENCE_RMS_THRESH)                   # NEW

                # whisper 解码
                segments, info = model.transcribe(
                    audio,
                    language=LANGUAGE,
                    beam_size=BEAM_SIZE,
                    vad_filter=USE_VAD,
                    vad_parameters=dict(min_silence_duration_ms=200),
                    condition_on_previous_text=False,
                    initial_prompt=None,
                    word_timestamps=False,
                )

                seg_texts = []
                seg_ts = []
                for seg in segments:
                    seg_texts.append(seg.text)
                    seg_ts.append((seg.start, seg.end, seg.text))

                partial_text = "".join(seg_texts).strip()

                # 去抖：只在变化时下发 partial
                if partial_text and partial_text != sess.last_partial_text:
                    sess.last_partial_text = partial_text
                    await ws.send_text(json.dumps({
                        "type": "partial",
                        "text": partial_text,
                        "avg_logprob": getattr(info, "avg_logprob", None),
                        "language": getattr(info, "language", None),
                    }))

                # === 端点器决定是否最终化（NEW） ===
                should_final, final_text = ep.update(
                    now_ms=now_ms,
                    partial_text=partial_text,
                    is_silence=is_silence,
                    tick_ms=tick_ms
                )

                if should_final and final_text:
                    await ws.send_text(json.dumps({
                        "type": "final",
                        "text": final_text,
                        "segments": [{"start": s, "end": e, "text": t} for (s, e, t) in seg_ts],
                    }))
                    asyncio.create_task(push_to_webhook(final_text))  # 不阻塞 ASR
                    sess.last_partial_text = ""  # final 后清空去抖
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
