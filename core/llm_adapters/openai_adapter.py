from openai import OpenAI
from typing import Iterator
from core.my_exception import InvokeException
from core.llm_adapters.base_adapter import BaseLLMAdapter
PROVIDER_THINKING_MAP = {
    "qwen" : {"enable_thinking":True},
    "deepseek" : {"thinking": {"type": "enabled"}}
}
class OpenaiLLMAdapter(BaseLLMAdapter):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        max_tokens:int,
        provider: str,
        temperature: float = 0.7
    ):
        super().__init__(model_name, api_key, base_url, max_tokens, provider, temperature)
        self._client = self._creat_client()
    def _creat_client(self):
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    def invoke(self, messages: list[dict[str, str]], enable_thinking = False, **kwargs) -> str:
        """
        非流式调用LLM，返回完整响应。
        适用于不需要流式输出的场景。
        """
        if enable_thinking:
            if self.provider == None:
                extra_body = kwargs.get("extra_body")
            else:
                extra_body=PROVIDER_THINKING_MAP[self.provider]
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=kwargs.get('temperature', self.temperature),
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    extra_body=extra_body,
                    **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
                )
                return "<think>\n" + \
                    response.choices[0].message.reasoning_content + \
                    "\n<\\think>\n" + \
                    response.choices[0].message.content
            except Exception as e:
                raise InvokeException(f"LLM调用失败: {str(e)}")
        else:
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=kwargs.get('temperature', self.temperature),
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
                )
                return response.choices[0].message.content
            except Exception as e:
                raise InvokeException(f"LLM调用失败: {str(e)}")
    
    def stream_invoke(self, messages: list[dict[str, str]], enable_thinking = False, **kwargs)->Iterator[str]:
        if enable_thinking:
            if self.provider == None:
                extra_body = kwargs.get("extra_body")
            else:
                extra_body=PROVIDER_THINKING_MAP[self.provider]
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    extra_body=extra_body,
                    **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
                )

                # 处理流式响应
                # print("✅ 大语言模型响应成功:")
                think_begin = True
                think_end = True
                is_answering = False
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                        if not is_answering:
                            if think_begin == True:
                                think_begin = False
                                yield "<think>\n"+delta.reasoning_content
                            else:
                                yield delta.reasoning_content
                    if hasattr(delta, "content") and delta.content:
                        if not is_answering:
                            is_answering = True
                            if think_end == True:
                                think_end = False
                                yield "\n<\\think>\n"+delta.content
                        else:
                            yield delta.content
            except Exception as e:
                print(f"❌ 调用LLM API时发生错误: {e}")
                raise InvokeException(f"LLM调用失败: {str(e)}")
        else:
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
                )

                # 处理流式响应
                # print("✅ 大语言模型响应成功:")
                for chunk in response:
                    if chunk.choices:
                        content = chunk.choices[0].delta.content or ""
                    if content:
                        # print(content, end="", flush=True)
                        yield content
                # print()  # 在流式输出结束后换行

            except Exception as e:
                print(f"❌ 调用LLM API时发生错误: {e}")
                raise InvokeException(f"LLM调用失败: {str(e)}")
    