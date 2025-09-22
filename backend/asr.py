"""
ASR 模块：语音识别
使用 faster-whisper 将音频转为文本
"""

from faster_whisper import WhisperModel

# 初始化模型（可以换成 large-v2 / medium，注意显存占用）
asr_model = WhisperModel("small", device="cuda", compute_type="float16")

def transcribe(audio_path: str) -> str:
    """
    输入：音频文件路径
    输出：识别的文本
    """
    segments, info = asr_model.transcribe(audio_path, beam_size=5)
    text = "".join([seg.text for seg in segments])
    return text.strip()
