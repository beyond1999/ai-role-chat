# pip install aiosqlite
import aiosqlite, time, uuid, json
from fastapi import FastAPI, Request
import uvicorn

DB_PATH = "conversations.db"
app = FastAPI()

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
          id TEXT PRIMARY KEY,
          created_at REAL
        );""")
        await db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
          id TEXT PRIMARY KEY,
          session_id TEXT,
          role TEXT CHECK(role IN ('user','assistant','system')),
          text TEXT,
          created_at REAL,
          idem TEXT UNIQUE,
          meta TEXT
        );""")
        await db.commit()

@app.on_event("startup")
async def _startup():
    await init_db()

async def save_message(session_id: str, role: str, text: str, idem: str|None, meta: dict|None=None):
    async with aiosqlite.connect(DB_PATH) as db:
        # 确保 session 存在
        await db.execute("INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?,?)",
                         (session_id, time.time()))
        try:
            await db.execute("""INSERT INTO messages
                (id, session_id, role, text, created_at, idem, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), session_id, role, text, time.time(),
                 idem, json.dumps(meta or {})))
            await db.commit()
            return True
        except aiosqlite.IntegrityError:
            # idem 冲突 → 认为是重复，忽略
            return False

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

    stored = await save_message(session_id, "user", text, idem, meta)
    print(f"[LLM] 收到文本: {text}  存储: {stored}")
    return {"status": "ok", "stored": stored}
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False, workers=1)
