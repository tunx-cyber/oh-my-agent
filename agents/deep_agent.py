"""
DeepAgent —— 自主 Agent，LLM 自己决定每一步做什么。

架构（教育目的，理解 Agent 如何自主工作）:

   用户: "帮我搜索论文并写报告"
     │
     ▼
   ┌──────────────────────────────────────┐
   │           AutoLoop 循环               │
   │                                      │
   │   ContextManager 构建上下文            │
   │        │                             │
   │        ▼                             │
   │   LLM 收到: system + tools + 历史     │
   │        │                             │
   │        ├─ "简单任务，直接回答" ──→ 返回 │
   │        │                             │
   │        ├─ "需要搜索" → web_search()   │
   │        │     ↓                       │
   │        │  观察结果 → 继续思考          │
   │        │     ↓                       │
   │        │  "还需要写报告" → write_file()│
   │        │     ↓                       │
   │        │  "写完了，总结给用户" → 返回   │
   │        │                             │
   │        └─ "不确定对不对" → 自己检查    │
   │             ↓                       │
   │          修正 → 继续                  │
   └──────────────────────────────────────┘
     │
     ▼
   最终回答（流式输出）

与固定流水线的区别:
  旧: planner → executor → reflector → router（每步强制）
  新: LLM 自由选择每一步（回答 | 工具 | 思考 | 检查）

Memory 和 Todo 追踪在后台自动运行，不干预 LLM 决策。
"""

from __future__ import annotations
from typing import Callable

from core.llm_mind import LLMMind
from core.state import StreamEvent
from core.agent_loop import AutoLoop
from core.context import ContextManager
from core.memory import MemoryManager
from core.todo import TodoManager
from tools.base import ToolRegistry


class DeepAgent:
    """
    自主 Agent —— LLM 自己决定是否需要规划、调用工具、反思结果。

    用法:
        llm = LLMMind(model_name="gpt-4o", ...)
        agent = DeepAgent(name="assistant", llm=llm, tools=[...])
        answer = agent.run("搜索最新AI论文")

        # 简单任务（无工具、无规划）
        answer = agent.run("什么是机器学习？")

        # 自定义回调
        agent.run("...", on_event=my_callback)
    """

    def __init__(
        self,
        name: str,
        llm: LLMMind,
        tools: list | None = None,
        system_prompt: str = "",
        max_iterations: int = 20,
        max_retries: int = 3,
        context_tokens: int = 8000,
    ):
        self.name = name
        self.llm = llm

        # ── 工具 ──────────────────────────────────
        self.tool_registry = ToolRegistry()
        if tools:
            for t in tools:
                self.tool_registry.register(t)

        # ── 核心模块 ──────────────────────────────
        self.context_manager = ContextManager(max_tokens=context_tokens)
        self.memory = MemoryManager(importance_threshold=0.5)
        self.todos = TodoManager()

        # ── 配置 ──────────────────────────────────
        self.system_prompt = system_prompt or self._build_system_prompt()
        self.max_iterations = max_iterations
        self.max_retries = max_retries

    # ── 主入口 ────────────────────────────────────
    def run(self, user_query: str, stream: bool = True, on_event: Callable | None = None) -> str:
        """
        运行自主循环。

        Args:
            user_query: 用户输入
            stream: 是否流式输出
            on_event: 自定义事件回调，None 时使用默认回调
        """
        # 重置
        self.memory.clear()
        self.todos = TodoManager()

        # 回调
        callback = on_event
        if callback is None and stream:
            callback = self._default_callback()

        # 发送开始事件
        if callback:
            callback(StreamEvent(type="start", content=user_query))

        # ── 构建上下文 ────────────────────────────
        memory_text = self.memory.summarize_for_context()
        todo_text = self.todos.format_for_context()

        messages = self.context_manager.build(
            system_prompt=self.system_prompt,
            user_query=user_query,
            memories=memory_text,
            todos=todo_text,
        )

        # ── 运行自主循环 ──────────────────────────
        loop = AutoLoop(
            llm=self.llm,
            tool_registry=self.tool_registry,
            on_event=callback,
            max_retries=self.max_retries,
        )

        # 用 stream_invoke 方式获取最终答案（流式）
        # AutoLoop.run() 内部处理工具调用，返回最终文本
        final_answer = loop.run(
            system_prompt=self.system_prompt,
            user_query=user_query,
            messages=messages,
            max_iterations=self.max_iterations,
            memory=self.memory,
            todos=self.todos,
        )

        # 如果 AutoLoop 返回了文本答案，但我们需要流式输出它
        # 这里做个权衡：AutoLoop 的 tool calls 期间已经流式了事件，
        # 最终答案我们直接输出

        # 发送完成事件
        if callback:
            callback(StreamEvent(type="complete", content=final_answer))

        # 记忆
        self.memory.remember(
            f"任务完成: {user_query[:100]}",
            kind="lesson", importance=0.8,
        )

        return final_answer

    # ── 系统提示 ─────────────────────────────────
    def _build_system_prompt(self) -> str:
        tools_desc = self.tool_registry.describe()
        return f"""你是 {self.name}，一个自主 AI Agent。

## 核心原则: 你决定每一步做什么

**简单问题** — 直接回答，不用工具
  例: 用户问"什么是Python" → 直接解释

**需要信息** — 调用工具获取，然后回答
  例: 用户问"当前目录有什么" → list_directory() → 回答

**复杂任务** — 分步思考，逐步执行
  1. 先想清楚需要做什么
  2. 逐步调用工具（每次一步）
  3. 检查结果是否合理
  4. 合成完整答案

**出错时** — 分析原因，修正重试
  例: 工具返回错误 → 检查参数 → 修正后再试

## 可用工具
{tools_desc or '(无 — 纯文本对话)'}

## 注意事项
- 每次只做一个工具调用（不要一次调用多个不相关的工具）
- 工具返回结果后，思考下一步该做什么
- 信息足够时立即给出最终答案
- 不确定时主动检查
"""

    # ── 默认流式回调 ─────────────────────────────
    def _default_callback(self) -> Callable[[StreamEvent], None]:
        state = {"in_stream": False}

        def cb(event: StreamEvent):
            t = event.type

            if t == "start":
                print(f"\n╭{'─'*58}╮")
                print(f"│ 🚀 {self.name}{' ' * max(1, 53 - len(self.name))}│")
                print(f"╰{'─'*58}╯")
                if event.content:
                    print(f"  📝 {event.content[:100]}")

            elif t == "tool_start":
                if state["in_stream"]:
                    print()
                    state["in_stream"] = False
                args = ", ".join(
                    f"{k}={str(v)[:30]}" for k, v in (event.tool_args or {}).items()
                )
                print(f"\n  🔧 {event.tool_name}({args})", end="", flush=True)

            elif t == "tool_end":
                icon = "✗" if not event.success else "✓"
                preview = (event.content or "")[:150].replace("\n", " ")
                print(f" → {icon} {preview}")

            elif t == "text":
                if not state["in_stream"]:
                    print(f"\n  ", end="", flush=True)
                    state["in_stream"] = True
                print(event.content, end="", flush=True)

            elif t == "text_done":
                if state["in_stream"]:
                    print()
                    state["in_stream"] = False

            elif t == "reasoning":
                if not state["in_stream"]:
                    print(f"\n  💭 ", end="", flush=True)
                    state["in_stream"] = True
                print(event.content, end="", flush=True)

            elif t == "reasoning_done":
                if state["in_stream"]:
                    print()
                    state["in_stream"] = False

            elif t == "error":
                print(f"\n  ❌ {event.content[:200]}")

            elif t == "complete":
                if state["in_stream"]:
                    print()
                    state["in_stream"] = False
                preview = (event.content or "")[:200].replace("\n", " ")
                print(f"\n╭{'─'*58}╮")
                print(f"│ ✅ 完成{' ' * 51}│")
                print(f"╰{'─'*58}╯")

        return cb
