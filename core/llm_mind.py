from typing import Literal, Optional, Iterator
import os
from dotenv import load_dotenv
PROVIDER_BASEURL_MAP = {
    "openai"   : "https://api.openai.com/v1",
    "deepseek" : "https://api.deepseek.com",
    "vllm"     : "http://localhost:8000/v1",
    "qwen"     : "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "ollma"    : "http://localhost:11434/v1"
}

class LLMMind:
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
        self._llm_dapter = self._create_llm_adapter()
    
    def _create_llm_adapter(self):
        if self.provider in ["openai", "vllm", "qwen", "ollama","deepseek"]:
            from core.llm_adapters.openai_adapter import OpenaiLLMAdapter
            return OpenaiLLMAdapter(
                self.model_name,
                self.api_key,
                self.base_url,
                self.max_tokens,
                self.provider,
                self.temperature
            )
        else:
            print("功能持续支持中")
            pass
    
    def invoke(self, messages: list[dict[str, str]], enable_thinking = False, **kwargs) -> str:
        return self._llm_dapter.invoke(
            messages=messages,
            enable_thinking=enable_thinking,
            **kwargs
        )
    
    def stream_invoke(self, messages: list[dict[str, str]], enable_thinking = False, **kwargs)->Iterator[str]:
        yield from self._llm_dapter.stream_invoke(
            messages=messages,
            enable_thinking=enable_thinking,
            **kwargs
        )
