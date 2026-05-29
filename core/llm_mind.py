"""
LLM 客户端 —— 直接基于 openai.OpenAI，无适配器抽象层。

支持:
- invoke():        非流式调用，返回完整文本
- stream_invoke(): 流式调用，yield 文本片段
- chat_json():     调用 + 自动解析 JSON 响应
- chat_with_tools(): 自动 tool-call 循环（function calling）
- 多 provider: openai / deepseek / qwen / vllm / ollama
- 思考模式: deepseek (thinking) / qwen (enable_thinking)
"""

from __future__ import annotations
import json
from typing import Iterator
from openai import OpenAI


PROVIDER_THINKING_MAP = {
    "qwen":     {"enable_thinking": True},
    "deepseek": {"thinking": {"type": "enabled"}},
}


class LLMMind:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        max_tokens: int = 4096,
        provider: str = "openai",
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.provider = provider
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    # ── 基础调用 ──────────────────────────────────
    def invoke(
        self,
        messages: list[dict[str, str]],
        enable_thinking: bool = False,
        **kwargs,
    ) -> str:
        """非流式调用，返回完整响应文本"""
        extra_body = None
        if enable_thinking and self.provider in PROVIDER_THINKING_MAP:
            extra_body = PROVIDER_THINKING_MAP[self.provider]

        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.pop("temperature", self.temperature),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            extra_body=extra_body,
            **kwargs,
        )

        msg = resp.choices[0].message
        if enable_thinking and hasattr(msg, "reasoning_content") and msg.reasoning_content:
            return f"<think>\n{msg.reasoning_content}\n</think>\n{msg.content or ''}"
        return msg.content or ""

    # ── 流式调用 ──────────────────────────────────
    def stream_invoke(
        self,
        messages: list[dict[str, str]],
        enable_thinking: bool = False,
        **kwargs,
    ) -> Iterator[str]:
        """流式调用，yield 文本片段"""
        extra_body = None
        if enable_thinking and self.provider in PROVIDER_THINKING_MAP:
            extra_body = PROVIDER_THINKING_MAP[self.provider]

        stream = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.pop("temperature", self.temperature),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            stream=True,
            extra_body=extra_body,
            **kwargs,
        )

        if enable_thinking:
            think_open = False
            think_close = False
            answering = False
            for chunk in stream:
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if not answering:
                        if not think_open:
                            think_open = True
                            yield f"<think>\n{delta.reasoning_content}"
                        else:
                            yield delta.reasoning_content
                if hasattr(delta, "content") and delta.content:
                    if not answering:
                        answering = True
                        if not think_close:
                            think_close = True
                            yield f"\n</think>\n{delta.content}"
                    else:
                        yield delta.content
        else:
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield chunk.choices[0].delta.content

    # ── JSON 模式 ─────────────────────────────────
    def chat_json(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> dict:
        """调用 LLM 并解析 JSON 响应"""
        raw = self.invoke(messages, **kwargs)

        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        return json.loads(raw)

    # ── 工具调用循环 ──────────────────────────────
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor: callable,
        max_rounds: int = 5,
        **kwargs,
    ) -> str:
        """
        自动 tool-call 循环: LLM → tool call → 结果回填 → LLM → ...

        tools: OpenAI function calling schema 列表
        tool_executor: (tool_name: str, tool_args: dict) -> str
        """
        msgs = list(messages)  # shallow copy

        for _ in range(max_rounds):
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=msgs,
                tools=tools,
                tool_choice="auto",
                temperature=kwargs.pop("temperature", self.temperature),
                max_tokens=kwargs.pop("max_tokens", self.max_tokens),
                **kwargs,
            )

            msg = resp.choices[0].message

            if not msg.tool_calls:
                return msg.content or ""

            # 回填 assistant 消息
            msgs.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # 执行工具并回填结果
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    result = tool_executor(tc.function.name, args)
                except Exception as e:
                    result = f"[Tool Error] {e}"
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        raise RuntimeError("Tool call loop exceeded max rounds")
