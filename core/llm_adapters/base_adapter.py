from openai import OpenAI
from typing import Iterator
from core.my_exception import InvokeException
PROVIDER_THINKING_MAP = {
    "qwen" : {"enable_thinking":True},
    "deepseek" : {"thinking": {"type": "enabled"}}
}
class BaseLLMAdapter:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        max_tokens:int,
        provider: str,
        temperature: float = 0.7
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.provider = provider
        self.temperature = temperature
        self._client = None
    def invoke(self, messages: list[dict[str, str]], enable_thinking = False, **kwargs) -> str:
        """
        非流式调用LLM，返回完整响应。
        适用于不需要流式输出的场景。
        """
        pass
    
    def stream_invoke(self, messages: list[dict[str, str]], enable_thinking = False, **kwargs)->Iterator[str]:
        pass

    def _creat_client(self):
        pass
    