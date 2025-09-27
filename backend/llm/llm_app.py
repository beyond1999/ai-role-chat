# llm.py
from fastapi import FastAPI, Request
import uvicorn

LLM_Module_PORT = 8001
LLM_Module_HOST = "127.0.0.1"
app = FastAPI()

# @app.post("/receive_text")
# async def receive_text(request: Request):
#     data = await request.json()
#     text = data.get("text", "")
#     print(f"[LLM 模块] 收到文本: {text}")
#     return {"status": "ok", "received": text}

@app.post("/receive_text")
async def receive_text(req: Request):
    data = await req.json()
    text = (data.get("text") or "").strip()
    session_id = data.get("session_id") or "default"
    idem = req.headers.get("X-Idempotency-Key")  # 可选幂等键
    meta = {
        "source": "asr",
        "language": data.get("language"),
        "avg_logprob": data.get("avg_logprob"),
        "segments": data.get("segments"),
    }
    if not text:
        return {"status": "empty"}

    # stored = await save_message(session_id, "user", text, idem, meta)
    print(f"session_id:{session_id}, text: {text}, idem: {idem}, meta: {meta}")
    stored = None
    print(f"[LLM] 收到文本: {text}  存储: {stored}")
    return {"status": "ok", "stored": stored}
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False, workers=1)




if __name__ == "__main__":
    # 启动在 http://127.0.0.1:8001
    uvicorn.run(app, host= LLM_Module_HOST, port=LLM_Module_PORT)
