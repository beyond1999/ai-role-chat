#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# uvicorn tts_xtts_server:app --host 0.0.0.0 --port 8002
import os
from pathlib import Path
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile

# 关键：Coqui XTTS v2（零样本克隆）
from TTS.api import TTS

# ============ 配置 ============
from pathlib import Path
REF_DIR = Path(__file__).resolve().parents[1] / "data" / "sound"   # backend/data/sound
# REF_DIR = Path(os.environ.get("XTTS_REF_DIR", "./refs"))  # 放参考音频的目录
DEVICE  = "cuda" if os.environ.get("XTTS_DEVICE","cuda") == "cuda" else "cpu"

# 人设 -> 参考音频 + 语言
PERSONA_MAP = {
    "wukong": {"ref": REF_DIR / "wukong_ref.wav", "lang": "zh"},
    "harry":  {"ref": REF_DIR / "harry_ref.wav",  "lang": "en"},
    "ironman":{"ref": REF_DIR / "ironman_ref.wav","lang": "en"},
}

# 只加载一次模型（首次运行会自动下载权重到 ~/.local/share/tts）
# 模型名随 TTS 版本可能略有差异；这个是常用别名：
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(MODEL_NAME).to(DEVICE)

# ============ FastAPI ============
app = FastAPI(title="XTTS v2 TTS", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class TTSIn(BaseModel):
    text: str
    persona: str | None = None
    language: str | None = None   # 可覆盖语言，如 "zh"/"en"
    # 可选：语速/风格等（XTTS 暂不支持 length_scale，但可用 prosody tokens/情绪等进阶玩法）

def _ensure_file(p: Path):
    if not p.exists():
        raise HTTPException(status_code=500, detail=f"reference wav missing: {p}")

@app.post("/tts", response_class=Response)
def tts_endpoint(req: TTSIn):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    persona = (req.persona or "wukong").strip()
    cfg = PERSONA_MAP.get(persona) or PERSONA_MAP["wukong"]

    ref_wav = cfg["ref"]
    _ensure_file(ref_wav)

    language = (req.language or cfg["lang"] or "zh").lower()

    with tempfile.TemporaryDirectory() as td:
        out_wav = Path(td) / "out.wav"
        # 关键：零样本克隆
        tts.tts_to_file(
            text=text,
            file_path=str(out_wav),
            speaker_wav=str(ref_wav),
            language=language
        )
        data = out_wav.read_bytes()

    return Response(content=data, media_type="audio/wav")

@app.get("/voices")
def voices():
    return {
        "device": DEVICE,
        "model": MODEL_NAME,
        "personas": {k: {"ref": str(v["ref"]), "lang": v["lang"]} for k, v in PERSONA_MAP.items()}
    }
