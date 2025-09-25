from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid

from backend.asr.asr import transcribe
from chat import chat_with_role
from tts import synthesize

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Role Chat Backend is running"}




app = FastAPI(title="AI Role Chat Backend")

# 音频临时保存目录
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/asr")
async def asr_endpoint(file: UploadFile):
    """
    上传音频文件 → 返回识别的文本
    """
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    text = transcribe(temp_path)
    return {"text": text}


@app.post("/chat")
async def chat_endpoint(
    user_text: str = Form(...),
    persona: str = Form("你是一个友好的助手")
):
    """
    输入文本 + 人设提示 → 返回模型回复
    """
    reply = chat_with_role(user_text, persona)
    return {"reply": reply}


@app.post("/tts")
async def tts_endpoint(text: str = Form(...)):
    """
    输入文本 → 返回音频文件
    """
    out_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")
    synthesize(text, out_path)
    return FileResponse(out_path, media_type="audio/wav", filename="reply.wav")


@app.get("/")
def root():
    return JSONResponse({"message": "AI Role Chat Backend is running!"})
