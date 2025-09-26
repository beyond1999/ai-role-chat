from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os, shutil, subprocess, threading
from pathlib import Path


import logging
from fastapi.responses import Response, JSONResponse

app = FastAPI(title="Piper TTS Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:63342", "http://127.0.0.1:63342",
        "http://localhost:8000",  "http://127.0.0.1:8000",
    ],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
logger = logging.getLogger("uvicorn.error")
TTS_PRESETS = {
    "cn_female_clear": {"voice":"zh-CN-XiaoxiaoNeural","rate":"+0%","pitch":"+0Hz"},
    "cn_male_calm":    {"voice":"zh-CN-YunxiNeural",   "rate":"-5%","pitch":"-1Hz"},
}
# --- 配置：可用环境变量覆盖 ---
PIPER_BIN  = os.getenv("PIPER_BIN", "piper")  # 已加入 PATH 则用命令名
# 默认模型（按你的实际路径修改为 WSL 路径）
PIPER_MODEL = os.getenv("PIPER_MODEL", "../../models/sound/voice-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx")
PIPER_CFG   = os.getenv("PIPER_CFG",   "../../models/sound/voice-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx.json")

def _check_ready(model: str, cfg: str):
    if shutil.which(PIPER_BIN) is None:
        raise HTTPException(500, f"Piper binary not found: {PIPER_BIN}")
    if not Path(model).exists() or not Path(cfg).exists():
        raise HTTPException(500, f"Piper model or config not found: {model} / {cfg}")

@app.get("/health")
def health():
    ok = shutil.which(PIPER_BIN) is not None and Path(PIPER_MODEL).exists() and Path(PIPER_CFG).exists()
    print(ok)
    return {"ok": ok}




@app.post("/tts_piper")
def tts_piper_post(
    text: str = Body(..., embed=True),
    model: str | None = Body(None),
    cfg:   str | None = Body(None),
):
    mdl = model or PIPER_MODEL
    cfgp = cfg   or PIPER_CFG
    _check_ready(mdl, cfgp)

    cmd = [
        PIPER_BIN,
        "--model", mdl,
        "--config", cfgp,
        "--output_file", "-",
        "--text", text,
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def _pump_err():
        for line in iter(p.stderr.readline, b""):
            logger.error("PIPER: %s", line.decode("utf-8", "ignore").rstrip())
    threading.Thread(target=_pump_err, daemon=True).start()

    def gen():
        for chunk in iter(lambda: p.stdout.read(4096), b""):
            yield chunk
        p.wait()

    return StreamingResponse(gen(), media_type="audio/wav")