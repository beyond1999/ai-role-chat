# AI Role Chat (MVP)

### 演示视频地址：
https://www.bilibili.com/video/BV1i1nZz8Ert/?spm_id_from=333.1387.homepage.video_card.click&vd_source=a2a1ad78786d36c0c9607eb4bfe43853

**语音 → ASR → LLM → TTS 的实时数字人最小可用系统**

- 前端：单页 HTML（选择人设：**孙悟空 / 哈利·波特 / 钢铁侠**，带头像与开场白）
- ASR：`FastAPI + WebSocket + faster-whisper`（端点检测、去重、低延迟）
- LLM：`FastAPI` 代理到 **llama.cpp/llama-server**（Qwen2.5 GGUF），OpenAI 兼容或 `/completion` 回退
- TTS：浏览器 `SpeechSynthesis`（占位；可替换为 `edge-tts` 服务）

---

## 目录
- [架构](#架构)
- [快速开始](#快速开始)
- [前端](#前端)
- [ASR 后端](#asr-后端)
- [LLM 模块](#llm-模块)
- [消息与协议](#消息与协议)
- [配置项](#配置项)
- [常见问题](#常见问题)
- [Roadmap](#roadmap)

---

## 架构
```
┌──────────┐   PCM(16k/Int16)   ┌──────────────┐   partial/final   ┌───────────┐
│  前端页面 │ ───────────────▶ │  ASR(WebSocket)│ ───────────────▶ │  前端页面  │
│  (麦克风) │ ◀─────────────── │  faster-whisper │ ◀───────────────│  (展示)   │
└──────────┘                   └──────────────┘                    └─────┬─────┘
     │  选人设+文本  POST /llm                                          │ TTS     │
     └───────────────────────────────────────────────────────────────▶│         │
                                                                      └──────────
                                    ┌──────────────────────────┐
                                    │ LLM 模块(FastAPI)       │  
     前端 POST /llm ───────────────▶│ 代理 llama-server       │──▶ llama.cpp `llama-server`
                                    │ (OpenAI /completion)    │   (Qwen2.5 GGUF)
                                    └──────────────────────────┘
```
> 说明：目前由**前端**在收到 ASR 的 `final` 后调用 `/llm`，ASR 端**不再**转发到 LLM（避免重复）。

---

## 快速开始

### 0) 依赖
- Python 3.10+
- 已编译好的 `llama.cpp/llama-server`（或使用你现有的二进制）
- 浏览器（推荐 Chrome/Edge）

```bash
pip install fastapi "uvicorn[standard]" httpx pydantic numpy aiohttp faster-whisper websockets
```

### 1) 启动本地 LLM（Qwen2.5 GGUF）
使用你现有脚本（示例）：
```bash
#!/usr/bin/env bash
set -e
MODEL=~/models/qwen2.5/3b-instruct/Qwen2.5-3B-Instruct-Q4_K_M.gguf
PORT=${PORT:-8080}
HOST=${HOST:-0.0.0.0}
CTX=${CTX:-4096}
TEMP=${TEMP:-0.7}
RP=${RP:-1.1}
GPU_LAYERS=${GPU_LAYERS:-35}
cd ~/work/llama.cpp/build/bin
./llama-server -m "$MODEL" --host "$HOST" --port "$PORT" \
  -c "$CTX" --temp "$TEMP" --repeat-penalty "$RP" --n-gpu-layers "$GPU_LAYERS"
```
> 端口缺省：`8080`。

### 2) 启动 LLM 模块（FastAPI 代理）
`backend/llm/llm_app.py`（已适配 llama.cpp OpenAI 兼容与 `/completion` 回退）：
```bash
export LLAMA_BASE=http://127.0.0.1:8080
python -m uvicorn backend.llm.llm_app:app --host 127.0.0.1 --port 8001 --workers 1 --reload false
```
> 健康检查：`curl http://127.0.0.1:8001/health`。

### 3) 启动 ASR 服务
`backend/asr/asr_app.py`：
```bash
# 注意：关闭 reload、多 worker，避免重复处理
python -m uvicorn backend.asr.asr_app:app --host 0.0.0.0 --port 8000 --workers 1 --reload false
```
> **WebSocket 路径**：你的代码是 `@app.websocket("/ws")`。若前端常量是 `/ws_asr`，请改成 `/ws` 或同步二者。

### 4) 打开前端
建议使用本地静态服务（麦克风权限在 `http://localhost` 下更稳定）：
```bash
# 在前端 HTML 所在目录
python -m http.server 5173
# 浏览器访问 http://127.0.0.1:5173
```
在页面里：选择**人设** → 点麦克风讲话 → 出现 `partial`/`final` → 自动发给 LLM → 浏览器朗读回复。

---

## 前端
文件：`AI Role Chat MVP (三人格+头像+开场对白).html`

- **人设**：
  - 孙悟空（🐵）  
  - 哈利·波特（🧙‍♂️）  
  - 钢铁侠（🤖）
- 切换人设会清空会话并发送**开场白**；`localStorage` 记住上次选择。
- **常量**：
  ```js
  const LLM_URL  = "http://127.0.0.1:8001/llm";
  const ASR_WS_URL = "ws://127.0.0.1:8000/ws_asr"; 
  ```
- **音频**：`AudioWorklet` 采集浮点 PCM → **前端重采样至 16kHz** → 转 Int16 → 20ms 一包通过 WS 发送。
- **TTS**：默认用浏览器 `speechSynthesis`（可替换后端 TTS）。

---

## ASR 后端
- 框架：`FastAPI` + `WebSocket`
- 模型：`faster-whisper`（CTranslate2），默认 `small`（可换 `medium/large-v2`）
- 核心：
  - **环形缓冲**：接收连续 PCM16；
  - **端点器**：RMS 静音判定 + 三规则（标点短停/长静音/文本稳定）；
  - **去重**：
    - 只在 `partial` 变化时下发；
    - `final` 后推进 **消费游标** `last_final_offset`，下一轮仅解码未消费音频；
    - 用 `last_final_text` 做一次幂等兜底。
- 重要参数：
  - `TICK_SECONDS=0.25`、`DECODE_WINDOW_SECONDS=8`
  - `END_SILENCE_MS=800`、`SHORT_PAUSE_MS=300`、`STABLE_NOCHANGE_MS=1500`
  - `SILENCE_RMS_THRESH≈0.005`（按麦克风底噪微调）
  - `LANGUAGE="zh"`（避免自动判成英文导致“thank you”）

> **不要**同时由 ASR 与前端都把 `final` 发给 LLM。当前版本已采用**前端发送**，请确保 ASR 里注释掉 `push_to_webhook(final_text)`。

---

## LLM 模块
- 框架：`FastAPI`，端口 `8001`
- 行为：将前端 `messages` 透传至 llama.cpp：
  - 优先 `POST /v1/chat/completions`（OpenAI兼容）
  - 若 404，回退 `POST /completion`（将 messages 拼成 prompt）
- 幂等：可传 `X-Idempotency-Key`，模块内存缓存 10 分钟，重复键复用首个结果。
- 环境变量：
  - `LLAMA_BASE`（默认 `http://127.0.0.1:8080`）
  - `LLAMA_MODEL`（标识用途）
  - `LLAMA_TIMEOUT`（默认 30s）

**示例请求**
```bash
curl -s http://127.0.0.1:8001/llm \
 -H 'Content-Type: application/json' \
 -H 'X-Idempotency-Key: demo-1' \
 -d '{
  "messages":[
    {"role":"system","content":"你是托尼·斯塔克风格：先结论后细节。"},
    {"role":"user","content":"给我两条苏州一日游建议"}
  ],
  "temperature":0.7,
  "max_tokens":256
 }' | jq .
```

---

## 消息与协议

### WebSocket（ASR）
- **连接**：`ws://<host>:8000/ws`
- **客户端 → 服务端**：
  1. 可选文本配置：`{"op":"config","sampleRate":16000}`
  2. 二进制帧：**PCM16**（单声道，16kHz），建议 **20ms/帧**
- **服务端 → 客户端**：
  - `{"type":"partial","text":"...","language":"zh"}`
  - `{"type":"final","text":"...","segments":[{"start":0.0,"end":1.2,"text":"..."}]}`

### HTTP（LLM）
- `POST /llm`
  - Body：`{ messages:[{role,content}...], temperature?, max_tokens?, session_id? }`
  - Header（可选）：`X-Idempotency-Key: <session:utter_idx:hash>`
  - 返回：`{"text":"...","model":"...","cached":false}`

---

## 配置项

### ASR 关键配置（`asr_app.py`）
- `MODEL_PATH`：faster-whisper 模型目录（或用名字让其自动下载）
- `WHISPER_DEVICE`：`cuda` 12.9 cudnn 12.9/`cpu`
- `WHISPER_COMPUTE_TYPE`：`float16`/`int8`/`int8_float16`
- `LANGUAGE`：建议固定为 `"zh"`
- 端点检测相关：`END_SILENCE_MS`、`SILENCE_RMS_THRESH` 等

### LLM 模块（`llm_app.py`）
- `LLAMA_BASE`、`LLAMA_TIMEOUT`、`LLAMA_MODEL`

### 前端（HTML）
- `LLM_URL`、`ASR_WS_URL`

