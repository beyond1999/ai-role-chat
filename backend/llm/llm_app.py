from __future__ import annotations
import os
import time
import json
from typing import List, Dict, Optional, Tuple

from fastapi import FastAPI, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import httpx

# ================== llama.cpp 服务配置 ==================
# 你的启动脚本：
# ./llama-server -m Qwen2.5-3B-Instruct-Q4_K_M.gguf --host 0.0.0.0 --port 8080 ...
# 这里默认按 OpenAI 兼容接口 /v1/chat/completions 调用；若不存在会自动回退到 /completion。
LLAMA_BASE = os.getenv("LLAMA_BASE", "http://127.0.0.1:8080")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "qwen2.5-3b-instruct")  # 仅用于标识；llama-server会忽略或覆盖
LLAMA_TIMEOUT = float(os.getenv("LLAMA_TIMEOUT", "30"))

# ================== 简易幂等缓存（内存） ==================
_IDEM: Dict[str, Tuple[float, Dict]] = {}
IDEM_TTL = 600.0  # 10 min

def _gc_idem():
    now = time.time()
    for k, (ts, _) in list(_IDEM.items()):
        if now - ts > IDEM_TTL:
            _IDEM.pop(k, None)

# ================== Pydantic 模型 ==================
class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    messages: List[Msg] = Field(..., description="OpenAI 风格 messages")
    session_id: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

class ChatResp(BaseModel):
    text: str
    model: str
    cached: bool = False

# ================== FastAPI 应用 ==================
app = FastAPI(title="LLM Module (llama.cpp client)", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "llama_base": LLAMA_BASE}

# ------------------ 调用适配 ------------------
async def _chat_via_openai_compat(req: ChatReq) -> Tuple[str, str]:
    url = f"{LLAMA_BASE}/v1/chat/completions"
    payload = {
        "model": LLAMA_MODEL,
        "messages": [m.model_dump() for m in req.messages],
        "temperature": req.temperature or 0.7,
        "max_tokens": req.max_tokens or 512,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=LLAMA_TIMEOUT) as client:
        r = await client.post(url, json=payload)
        if r.status_code == 404:
            raise FileNotFoundError("/v1/chat/completions not found")
        r.raise_for_status()
        data = r.json()
        text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        used_model = data.get("model", LLAMA_MODEL)
        return text, used_model


def _to_legacy_prompt(messages: List[Msg]) -> str:
    sys = next((m.content for m in messages if m.role == "system"), "")
    parts = []
    if sys:
        parts.append(f"System: {sys}")
    for m in messages:
        if m.role == "user":
            parts.append(f"User: {m.content}")
        elif m.role == "assistant":
            parts.append(f"Assistant: {m.content}")
    parts.append("Assistant:")
    return "".join(parts)

async def _chat_via_legacy_completion(req: ChatReq) -> Tuple[str, str]:
    # llama.cpp 旧接口 /completion
    url = f"{LLAMA_BASE}/completion"
    prompt = _to_legacy_prompt(req.messages)
    payload = {
        "prompt": prompt,
        "n_predict": req.max_tokens or 512,
        "temperature": req.temperature or 0.7,
        "cache_prompt": True,
    }
    async with httpx.AsyncClient(timeout=LLAMA_TIMEOUT) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        text = data.get("content") or data.get("generated_text") or data.get("text") or ""
        if isinstance(text, list):
            text = "".join(text)
        return (text or "").strip(), "llama.cpp:completion"

async def _chat_via_llama(req: ChatReq) -> Tuple[str, str]:
    try:
        return await _chat_via_openai_compat(req)
    except FileNotFoundError:
        return await _chat_via_legacy_completion(req)

# ------------------ HTTP 接口 ------------------
@app.post("/llm", response_model=ChatResp)
async def llm_endpoint(req: ChatReq, x_idempotency_key: Optional[str] = Header(None)):
    # 幂等缓存
    if x_idempotency_key:
        _gc_idem()
        cached = _IDEM.get(x_idempotency_key)
        if cached:
            return ChatResp(text=cached[1]["text"], model=cached[1]["model"], cached=True)

    try:
        text, model_used = await _chat_via_llama(req)
    except Exception as e:
        # 兜底（不抛 500），避免前端体验断裂
        text = f"[本地模型暂不可用] {type(e).__name__}: {e}"
        model_used = "llama.cpp:error"

    out = {"text": text, "model": model_used}
    if x_idempotency_key:
        _IDEM[x_idempotency_key] = (time.time(), out)
    return ChatResp(**out)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False, workers=1)
