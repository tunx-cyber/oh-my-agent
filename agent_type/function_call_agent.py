from typing import Any, Union

def _invoke_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: Union[str, dict], **kwargs):
        """调用底层OpenAI客户端执行函数调用"""
        client = getattr(self.llm, "_client", None)
        if client is None:
            raise RuntimeError("Agents未正确初始化客户端，无法执行函数调用。")

        client_kwargs = dict(kwargs)
        client_kwargs.setdefault("temperature", self.llm.temperature)
        if self.llm.max_tokens is not None:
            client_kwargs.setdefault("max_tokens", self.llm.max_tokens)

        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **client_kwargs,
        )

#内部逻辑是对Openai 原生的functioncall作再封装
#OpenAI 原生functioncall示例
from openai import OpenAI
client = OpenAI()

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
          },
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
      },
    }
  }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
completion = client.chat.completions.create(
  model="gpt-5",
  messages=messages,
  tools=tools,
  tool_choice="auto"
)

print(completion)
