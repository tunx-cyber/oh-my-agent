"""
工具系统 —— LangChain 风格的 Tool / ToolRegistry。

设计理念（教育目的）:
  工具是 Agent 的"手"——让 LLM 能够与外部世界交互。
  每个工具对外暴露 JSON Schema（供 LLM function calling 使用），
  对内封装实际的 Python 函数。

关键设计:
  1. Tool 封装: name + description + parameters_schema + func
  2. ToolRegistry: 注册、查找、执行、导出 schema
  3. @tool 装饰器: 一键将普通函数转为 Tool（类似 @langchain_core.tools.tool）
  4. to_openai_schema(): 生成 OpenAI function calling 兼容格式

用法:
    # 方式 1: 装饰器
    @tool("calculator", "执行数学计算")
    def calculator(expression: str) -> str:
        return str(eval(expression))

    # 方式 2: 直接构造
    my_tool = Tool(func=my_func, name="my_tool", description="...")

    # 注册 & 执行
    registry = ToolRegistry()
    registry.register(calculator)
    result = registry.call("calculator", expression="2+2")

    # 导出给 LLM
    schemas = registry.to_openai_schemas()
"""

from __future__ import annotations
from typing import Callable, Any
from dataclasses import dataclass, field
import inspect
import json


# ═══════════════════════════════════════════════════
# Tool 定义
# ═══════════════════════════════════════════════════

@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str = "string"       # string | integer | number | boolean | array | object
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[str] | None = None

    def to_json_schema(self) -> dict:
        prop: dict = {"type": self.type, "description": self.description}
        if self.enum:
            prop["enum"] = self.enum
        return prop


class Tool:
    """
    单个工具的封装。

    属性:
      name: 工具名（LLM 用此名调用）
      description: 工具描述（LLM 据此判断何时使用）
      parameters: 参数列表
      func: 实际执行的 Python 函数
    """

    def __init__(
        self,
        func: Callable | None = None,
        name: str | None = None,
        description: str | None = None,
        parameters: list[ToolParameter] | None = None,
    ):
        self.func = func
        self.name = name or (func.__name__ if func else "unnamed")
        self.description = description or self._extract_desc(func)
        self.parameters = parameters or self._parse_parameters(func)

    def __call__(self, **kwargs) -> Any:
        """让 Tool 实例可以像函数一样调用"""
        if self.func is None:
            raise RuntimeError(f"Tool '{self.name}' has no function bound")
        return self.func(**kwargs)

    def __repr__(self) -> str:
        return f"Tool(name='{self.name}', desc='{self.description[:40]}...')"

    # ── 参数解析（从函数签名自动推断） ──────────
    @staticmethod
    def _parse_parameters(func: Callable | None) -> list[ToolParameter]:
        if func is None:
            return []
        sig = inspect.signature(func)
        params = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            annotation = param.annotation
            param_type = Tool._python_type_to_str(annotation)
            required = param.default is inspect.Parameter.empty
            default = None if required else param.default
            params.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=f"参数 {param_name}",
                required=required,
                default=default,
            ))
        return params

    @staticmethod
    def _extract_desc(func: Callable | None) -> str:
        if func is None:
            return "No description"
        doc = inspect.getdoc(func)
        if doc:
            return doc.split("\n")[0].strip()
        return func.__name__

    @staticmethod
    def _python_type_to_str(annotation) -> str:
        if annotation is inspect.Parameter.empty:
            return "string"
        origin = getattr(annotation, "__origin__", None)
        if origin is list:
            return "array"
        if origin is dict:
            return "object"
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
        return type_map.get(annotation, "string")

    # ── Schema 导出 ───────────────────────────────
    def to_openai_schema(self) -> dict:
        """生成 OpenAI function calling 兼容的 tool schema"""
        properties = {}
        required = []

        for p in self.parameters:
            properties[p.name] = p.to_json_schema()
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {"name": p.name, "type": p.type, "description": p.description,
                 "required": p.required, "default": p.default}
                for p in self.parameters
            ],
        }


# ═══════════════════════════════════════════════════
# ToolRegistry —— 工具注册中心
# ═══════════════════════════════════════════════════

class ToolRegistry:
    """
    全局工具注册表。

    提供: 注册、查找、按标签过滤、批量执行、导出 schema。
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    # ── 注册 ──────────────────────────────────────
    def register(self, tool: Tool | Callable):
        """注册一个工具（Tool 实例或裸 Callable）"""
        if isinstance(tool, Tool):
            self._tools[tool.name] = tool
        elif callable(tool):
            t = Tool(func=tool)
            self._tools[t.name] = t
        else:
            raise TypeError(f"Expected Tool or Callable, got {type(tool)}")

    # ── 执行 ──────────────────────────────────────
    def call(self, name: str, **kwargs) -> Any:
        """按名称调用工具"""
        if name not in self._tools:
            available = ", ".join(self._tools.keys())
            raise KeyError(f"Unknown tool '{name}'. Available: {available}")
        return self._tools[name](**kwargs)

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    # ── Schema 导出（给 LLM） ──────────────────────
    def to_openai_schemas(self) -> list[dict]:
        """导出为 OpenAI function calling tools 列表"""
        return [t.to_openai_schema() for t in self._tools.values()]

    def describe(self) -> str:
        """生成人类可读的工具列表（用于 prompt）"""
        if not self._tools:
            return "(无可用工具)"
        lines = []
        for t in self._tools.values():
            param_str = ", ".join(
                f"{p.name}: {p.type}" + ("?" if not p.required else "")
                for p in t.parameters
            )
            lines.append(f"- {t.name}({param_str}): {t.description}")
        return "\n".join(lines)

    # ── 查询 ──────────────────────────────────────
    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def clear(self):
        self._tools.clear()

    # ── 向后兼容（旧 API） ─────────────────────────
    def register_tool(self, tool: Tool | Callable):
        """[兼容旧 API] 同 register()"""
        self.register(tool)

    def get_tool(self, name: str) -> Tool:
        """[兼容旧 API] 同 get()"""
        return self.get(name)

    def execute_tool(self, name: str, **params) -> Any:
        """[兼容旧 API] 同 call()"""
        return self.call(name, **params)

    def get_tools_description(self) -> str:
        """[兼容旧 API] 同 describe()"""
        return self.describe()

    def list_tools(self) -> list[str]:
        """[兼容旧 API] 同 names()"""
        return self.names()


# ═══════════════════════════════════════════════════
# @tool 装饰器 —— 一键创建 Tool
# ═══════════════════════════════════════════════════

def tool(name: str | None = None, description: str | None = None) -> Callable:
    """
    装饰器: 将普通函数一键转为 Tool。

    用法:
        @tool("search", "搜索互联网")
        def search(query: str, top_k: int = 5) -> str:
            ...

        @tool  # 自动从函数名和 docstring 推断
        def read_file(path: str) -> str:
            '''读取文件内容'''
            ...
    """

    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_desc = description
        if tool_desc is None:
            doc = inspect.getdoc(func)
            tool_desc = doc.split("\n")[0].strip() if doc else func.__name__
        return Tool(func=func, name=tool_name, description=tool_desc)

    return decorator


# ═══════════════════════════════════════════════════
# 全局默认 registry（可选）
# ═══════════════════════════════════════════════════
default_registry = ToolRegistry()
