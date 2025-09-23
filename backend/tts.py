"""
TTS 模块：文本转语音
这里用 Piper/Coqui TTS
"""

import subprocess
import os

# 假设你安装了 Piper，并下载了一个模型
PIPER_BIN = "piper"
VOICE_MODEL = "voices/zh_cn-voice.onnx"

def synthesize(text: str, out_path: str = "output.wav") -> str:
    """
    输入：文本
    输出：合成后的语音文件路径
    """
    cmd = [PIPER_BIN, "--model", VOICE_MODEL, "--output_file", out_path]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    process.communicate(input=text.encode("utf-8"))
    return out_path
