# 在能联网的机器/环境里执行一次
from faster_whisper import WhisperModel
model = WhisperModel("Systran/faster-whisper-small", download_root="../models")
