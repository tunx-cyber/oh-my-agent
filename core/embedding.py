"""
Embedding 客户端 —— 直接基于 openai.OpenAI，无适配器抽象层。
"""

from openai import OpenAI


class EmbeddingClient:
    def __init__(self, api_url: str, api_key: str, model: str):
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=api_url)

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """返回 embedding 向量"""
        is_single = isinstance(text, str)
        inputs = [text] if is_single else text

        resp = self._client.embeddings.create(input=inputs, model=self.model)
        vectors = [d.embedding for d in resp.data]
        return vectors[0] if is_single else vectors
