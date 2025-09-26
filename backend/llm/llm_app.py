# llm.py
from fastapi import FastAPI, Request
import uvicorn

LLM_Module_PORT = 8001
LLM_Module_HOST = "127.0.0.1"
app = FastAPI()

@app.post("/receive_text")
async def receive_text(request: Request):
    data = await request.json()
    text = data.get("text", "")
    print(f"[LLM 模块] 收到文本: {text}")
    return {"status": "ok", "received": text}

if __name__ == "__main__":
    # 启动在 http://127.0.0.1:8001
    uvicorn.run(app, host= LLM_Module_HOST, port=LLM_Module_PORT)
