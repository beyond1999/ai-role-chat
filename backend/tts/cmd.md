```command

uvicorn piper_tts_app:app --host 0.0.0.0 --port 8002 --reload
uvicorn edge_tts_app:app --host 0.0.0.0 --port 8001 --reload
```

```
XTTS_REF_DIR=./refs XTTS_DEVICE=cuda uvicorn tts_xtts_server:app --host 0.0.0.0 --port 8002
```


```
curl -s 'http://127.0.0.1:8002/voices'
curl -s -X POST 'http://127.0.0.1:8002/tts' \
  -H 'Content-Type: application/json' \
  -d '{"text":"俺老孙来也！", "persona":"wukong"}' \
  --output /tmp/wk.wav
# 播放 /tmp/wk.wav 试听

```


