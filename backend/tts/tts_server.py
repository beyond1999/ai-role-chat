#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cd backend/tts/
# uvicorn tts_server:app --host 0.0.0.0 --port 8002
import os
import tempfile
import subprocess
from pathlib import Path
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

"""
运行方式：
  uvicorn tts_server:app --host 0.0.0.0 --port 8002

准备工作：
1) 安装 piper 可执行程序（确保命令行能运行 `piper --help`）。
2) 下载语音模型到 VOICE_DIR 目录（见下面 VOICE_MAP 注释）。
   Piper 每个语音包含两个文件：xxx.onnx 和 xxx.onnx.json（或 .json）。
3) 按你的机器路径改 VOICE_DIR 和 VOICE_MAP。
"""

# ====== 你的语音库目录（可放多语种模型）======
VOICE_DIR = os.environ.get("PIPER_VOICE_DIR", "/mnt/d/github-repo/ai-role-chat/data/voice")

# ====== 人设 -> 语音模型文件名映射（示例）======
# 提示：你可以用 `ls VOICE_DIR` 看看有哪些可用的模型，再改下面文件名。
# 例：官方常见中文：zh_CN-huayan-medium；英文可用：en_US-lessac-medium、en_US-amy-medium 等（以你的库为准）
VOICE_MAP = {
    "wukong": {  # 孙悟空（中文）
        "model": "zh_CN-huayan-medium.onnx",
        "config": "zh_CN-huayan-medium.onnx.json",
        "length_scale": 1.0,   # 语速/停连（>1更慢，<1更快）
        "speaker_id": None,    # 多说话人模型可指定，如 0、1；无就 None
    },
    "harry": {   # 哈利·波特风（英文）
        "model": "en_US-lessac-medium.onnx",
        "config": "en_US-lessac-medium.onnx.json",
        "length_scale": 1.0,
        "speaker_id": None,
    },
    "ironman": { # 钢铁侠风（英文）
        "model": "en_US-amy-medium.onnx",
        "config": "en_US-amy-medium.onnx.json",
        "length_scale": 0.95,  # 略微干练
        "speaker_id": None,
    },
}

# ========== FastAPI ==========
app = FastAPI(title="TTS Service", version="0.1.0")

# 允许前端本地访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 如需更严格可改为你的前端源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSIn(BaseModel):
    text: str
    persona: str | None = None
    # 可选参数
    length_scale: float | None = None  # 全局覆盖语速/停连
    speaker_id: int | None = None      # 若模型支持多说话人
    # 兼容将来扩展（如：noise_scale、noise_w 等）
    # noise_scale: float | None = None
    # noise_w: float | None = None

def _ensure_file(p: Path):
    if not p.exists():
        raise FileNotFoundError(str(p))

def synthesize_with_piper(text: str, voice_cfg: dict) -> bytes:
    """
    使用 Piper CLI 合成，输出 wav 字节。
    做法：把 stdout 写入临时 wav 文件，再读取回来。
    （piper 版本不同，直出 stdout 的行为不一，这里走临时文件最稳。）
    """
    model = Path(VOICE_DIR) / voice_cfg["model"]
    config = Path(VOICE_DIR) / voice_cfg["config"]
    _ensure_file(model)
    _ensure_file(config)

    # 可选参数
    speaker_id = voice_cfg.get("speaker_id")
    length_scale = voice_cfg.get("length_scale", 1.0)

    # 生成临时 wav
    with tempfile.TemporaryDirectory() as td:
        out_wav = Path(td) / "out.wav"
        cmd = [
            "piper",
            "--model", str(model),
            "--config", str(config),
            "--length_scale", str(length_scale),
            "--output_file", str(out_wav),
        ]
        if speaker_id is not None:
            cmd += ["--speaker", str(speaker_id)]

        # 通过 stdin 输入文本
        try:
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Piper synthesis failed.\nSTDERR:\n{e.stderr.decode(errors='ignore')}"
            )

        if not out_wav.exists():
            raise RuntimeError("Piper did not produce output WAV.")

        data = out_wav.read_bytes()
        return data

@app.post("/tts", response_class=Response)
def tts(req: TTSIn):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    persona = (req.persona or "wukong").strip()
    if persona not in VOICE_MAP:
        # fallback：不存在的人设用 wukong
        persona = "wukong"

    # 合并用户覆盖参数
    base = VOICE_MAP[persona].copy()
    if req.length_scale is not None:
        base["length_scale"] = float(req.length_scale)
    if req.speaker_id is not None:
        base["speaker_id"] = int(req.speaker_id)

    try:
        wav_bytes = synthesize_with_piper(text, base)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Voice model missing: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=wav_bytes, media_type="audio/wav")

@app.get("/voices")
def list_voices():
    """简单列出你配置的人设->模型映射，便于前端可视化选择。"""
    return {
        "voice_dir": VOICE_DIR,
        "personas": {
            k: {"model": v["model"], "config": v["config"], "speaker_id": v.get("speaker_id")}
            for k, v in VOICE_MAP.items()
        }
    }
