import os
import asyncio
from typing import Optional

import httpx
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

# ========= 配置 =========
# OpenAI 兼容接口（也可指向本地 LLM 网关，如 vLLM/Ollama/One-API）
LLM_BASE = os.getenv("LLM_BASE", "http://localhost:9000")     # 末尾不要带斜杠
LLM_PATH = os.getenv("LLM_PATH", "/v1/chat/completions")
LLM_URL  = f"{LLM_BASE}{LLM_PATH}"
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")                    # 没有就留空

# 可选：把 LLM 回复转发给 TTS（文本→语音）
# 例如: http://tts-svc:7002/speak
TTS_WEBHOOK = os.getenv("TTS_WEBHOOK", "")                    # 留空则不转发

# 可选：也把 LLM 文本回发给前端管理进程（如果你有这样的 HTTP 入口）
# FRONTEND_WEBHOOK = os.getenv("FRONTEND_WEBHOOK", "")

TIMEOUT = float(os.getenv("LLM_HTTP_TIMEOUT", "60"))

# ========= 数据模型 =========
class IngestASR(BaseModel):
    text: str
    sid: Optional[str] = None       # 强烈建议从 ASR 传过来，便于路由
    meta: Optional[dict] = None     # 可携带语言/时间戳等

class LLMReply(BaseModel):
    sid: Optional[str] = None
    text: str

# ========= 应用 =========
app = FastAPI(title="LLM Service", version="0.1.0")

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/ingest_asr")
async def ingest_asr(payload: IngestASR, bg: BackgroundTasks):
    """
    ASR -> LLM 的入口。
    返回立即 200，真实的 LLM 处理放后台任务，不阻塞 ASR。
    """
    # 后台处理（LLM -> 可选 TTS）
    bg.add_task(_handle_asr_to_llm, payload)
    return {"ok": True}

# ========= 核心处理 =========
async def _handle_asr_to_llm(payload: IngestASR):
    """
    1) 调用 LLM 得到回复
    2) 可选地把回复推给 TTS 服务（带着 sid）
    """
    reply_text = await _call_llm(payload.text)

    # 如果需要先把 LLM 文本回给“中控/前端服务”，可在此 POST：
    # if FRONTEND_WEBHOOK:
    #     await _post_json(FRONTEND_WEBHOOK, {"sid": payload.sid, "text": reply_text, "type": "llm_text"})

    # 可选：把文字交给 TTS
    if TTS_WEBHOOK:
        await _post_json(TTS_WEBHOOK, {"sid": payload.sid, "text": reply_text})

async def _call_llm(user_text: str) -> str:
    """
    调用一个 OpenAI 兼容的 LLM Chat Completion 接口，返回字符串回复。
    如需流式，可改用 /stream 并逐块聚合；或直接把流转给 TTS（更复杂）。
    """
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    json_payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个实时语音对话助手，回答简洁准确。"},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.7,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(LLM_URL, headers=headers, json=json_payload)
        r.raise_for_status()
        data = r.json()
        # 兼容 OpenAI 风格
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            # 兜底：不同网关字段差异时
            return str(data)

async def _post_json(url: str, body: dict):
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
    except Exception as e:
        # 生产环境使用结构化日志
        print(f"[LLM] POST {url} failed: {e}")
