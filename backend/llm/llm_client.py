import httpx
from typing import List, Dict, AsyncIterator, Optional

class LlamaServerClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        model: str = "qwen2.5-3b-instruct-q4_k_m"
    ) -> str:
        """
        调用 /v1/chat/completions，返回完整回答
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            # 标准 OpenAI 风格：choices[0].message.content
            return data["choices"][0]["message"]["content"]

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        model: str = "qwen2.5-3b-instruct-q4_k_m"
    ) -> AsyncIterator[str]:
        """
        调用 /v1/chat/completions，流式返回
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as r:
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    if line.strip() == "data: [DONE]":
                        break
                    chunk = line[len("data: "):]
                    try:
                        obj = httpx.Response.json(r=None, text=chunk)  # 解析 JSON
                    except Exception:
                        continue
                    delta = obj["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
