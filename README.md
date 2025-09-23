# ai-role-chat
AI role-play voice chat demo for Qiniu Cloud 2026 campus recruiting



一个为七牛云 2026 校招挑战赛开发的 AI 角色扮演语音聊天系统，集成 ASR → LLM → TTS 的完整链路。

- **功能**：用户选择角色（中文影视或虚拟人物），说话 → AI 回答（语音），支持角色稳定性、人设记忆与对话打断。
- **亮点**：低延迟、中文角色、人设一致 + 技术指标监控，让系统更像真实角色陪伴体验。


- Python (FastAPI)
- Faster-Whisper (ASR)
- vLLM + Qwen 模型 (LLM)
- Piper / Coqui TTS (TTS)

## TODO
- [ ] 初始化后端框架
- [ ] 接入 ASR 模块
- [ ] 接入 LLM 模块
- [ ] 接入 TTS 模块
```commandline
# linux
source venv/bin/activate

# windows
venv/Script/activate

python -m http.server 8000
```