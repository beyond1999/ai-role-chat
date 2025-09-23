"""
Chat 模块：调用 vLLM / HuggingFace Transformers
生成角色化回复
"""

from transformers import AutoTokenizer
import requests

# 假设你用 vLLM 起了一个服务：http://localhost:8000/v1
VLLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def chat_with_role(user_text: str, persona_prompt: str = "") -> str:
    """
    输入：用户文本 + 人设提示
    输出：模型回复
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": user_text}
        ],
        "max_tokens": 300,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }

    response = requests.post(VLLM_ENDPOINT, json=payload, timeout=60)
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()
