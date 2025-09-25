# backend/tts/edge_tts_app.py
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import edge_tts

app = FastAPI(title="Edge TTS Service", version="0.1.0")


# 允许你的前端与其他后端来源（按需增减）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:63342", "http://127.0.0.1:63342",
        "http://localhost:8000",  "http://127.0.0.1:8000",
    ],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# 可根据偏好再加
TTS_PRESETS = {
    "cn_female_clear": {"voice": "zh-CN-XiaoxiaoNeural", "rate": "+0%", "pitch": "+0Hz"},
    "cn_male_calm":    {"voice": "zh-CN-YunxiNeural",    "rate": "-5%", "pitch": "-1Hz"},
    "iron_man_vibe":   {"voice": "en-US-GuyNeural",      "rate": "-5%", "pitch": "-1Hz"},
    "harry_vibe":      {"voice": "en-GB-RyanNeural",     "rate": "+5%", "pitch": "+2Hz"},
}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/tts_presets")
def tts_presets():
    return JSONResponse(TTS_PRESETS)

@app.post("/tts_edge_tts")
async def tts(
    text: str = Body(..., embed=True),
    voice: str = "zh-CN-XiaoxiaoNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
    preset: str | None = None,
):
    # 若传入预设则覆盖
    if preset and preset in TTS_PRESETS:
        p = TTS_PRESETS[preset]
        voice = p.get("voice", voice)
        rate  = p.get("rate", rate)
        pitch = p.get("pitch", pitch)

    # 用 SSML 让 rate/pitch/volume 生效
    ssml = f"""{text}""".strip()

    com = edge_tts.Communicate(ssml, voice=voice, volume=volume,pitch=pitch)

    async def gen():
        async for chunk in com.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    return StreamingResponse(gen(), media_type="audio/mpeg")
