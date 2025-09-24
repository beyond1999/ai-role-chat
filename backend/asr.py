"""
ASR 模块：语音识别
使用 faster-whisper 将音频转为文本
"""

from faster_whisper import WhisperModel
import os
os.environ.setdefault("CT2_USE_CUDNN", "0")
os.environ.setdefault("CTRANSLATE2_USE_CUDNN", "0")

# os.environ.setdefault("CT2_USE_CUDNN", "1")
# os.environ.setdefault("CTRANSLATE2_USE_CUDNN", "1")

os.environ.setdefault("CT2_CUDA_TRUE_FP16_GEMM", "0")
os.environ.setdefault("CTRANSLATE2_CUDA_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import time




# 初始化模型（可以换成 large-v2 / medium，注意显存占用）
asr_model = WhisperModel("../models/faster-whisper-small", device="cpu", compute_type="int8")
# asr_model = WhisperModel("../models/faster-whisper-small", device="cpu", compute_type="int8", local_files_only=True)
def transcribe(audio_path: str) -> str:
    """
    输入：音频文件路径
    输出：识别的文本
    """
    segments, info = asr_model.transcribe(audio_path, beam_size=1,       # ← 先小
                          vad_filter=False)
    text = "".join([seg.text for seg in segments])
    return text.strip()


def test_time(file_name):
    start = time.time()   # 记录开始时间

    text = transcribe("../data/sound/" + file_name)
    # hello.m4a
    # test.m4a
    print(text)

    end = time.time()     # 记录结束时间
    print(f"运行耗时: {end - start:.4f} 秒")

if __name__ == "__main__":

    file_name = input("please enter the file name:")
    while file_name != "exit" and file_name != "exit()":
        test_time(file_name)
        file_name = input("please enter the file name:")


    