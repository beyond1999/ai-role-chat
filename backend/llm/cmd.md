qwen
```
~/work/llama.cpp/build/bin/run_qwen_server.sh
```

```
#!/usr/bin/env bash
set -e
MODEL=~/models/qwen2.5/3b-instruct/Qwen2.5-3B-Instruct-Q4_K_M.gguf
PORT=${PORT:-8080}
HOST=${HOST:-0.0.0.0}
CTX=${CTX:-4096}
TEMP=${TEMP:-0.7}
RP=${RP:-1.1}
GPU_LAYERS=${GPU_LAYERS:-35}   # CPU就改成0

cd ~/work/llama.cpp/build/bin
./llama-server \
  -m "$MODEL" \
  --host "$HOST" --port "$PORT" \
  -c "$CTX" --temp "$TEMP" --repeat-penalty "$RP" \
  --n-gpu-layers "$GPU_LAYERS"

```

```
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b-instruct-q4_k_m",
    "messages": [
      {"role": "system", "content": "你是一个简洁、靠谱的中文助理。"},
      {"role": "user", "content": "你好，请用两三句话自我介绍，并告诉我你能做什么。"}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": false
  }'

```

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b-instruct-q4_k_m",
    "messages": [
      {"role": "system", "content": "你是喜羊羊。"},
      {"role": "user", "content": "你好，请用两三句话自我介绍，还有你了解懒羊羊吗。"}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": false
  }'