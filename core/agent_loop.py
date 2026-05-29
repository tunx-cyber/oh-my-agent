"""
Agent 循环引擎 —— LLM 自己决定每一步做什么。

设计理念（教育目的）:
  真正的 agent 不是被流水线推着走，而是 LLM 自己决定:
  - 简单问题 → 直接回答（不规划、不自省）
  - 需要信息 → 调用工具 → 观察结果 → 继续
  - 复杂任务 → 先想步骤 → 逐步执行 → 完成后检查
  - 出错了   → 修正重试

引擎只提供循环骨架，LLM 是驾驶员。

两种引擎:
  1. AutoLoop    — 自主循环（LLM 驱动，默认）
  2. AgentEngine — 固定流水线（plan→execute→reflect，教育展示用）
"""

from __future__ import annotations
import json
from typing import Callable, Iterator
from dataclasses import dataclass

from core.state import AgentState, RouteDecision


@dataclass
class EngineEvent:
    """引擎在节点完成后产出的事件"""
    node_name: str
    updates: dict
    state: AgentState
    route_decision: str | None = None


NodeFn = Callable[[AgentState], dict]
RouterFn = Callable[[AgentState], str]
END = "__END__"


# ═══════════════════════════════════════════════════
# AutoLoop —— 自主循环（默认）
# ═══════════════════════════════════════════════════

class AutoLoop:
    """
    自主 Agent 循环 —— LLM 自己决定每一步。

    与 AgentEngine 的区别:
      AgentEngine: 固定流水线 planner→executor→reflector→router
      AutoLoop:    LLM 自由选择: 回答 | 调工具 | 规划 | 反思

    用法:
        loop = AutoLoop(llm, tools, on_event=callback)
        answer = loop.run(
            system_prompt="你是 AI 助手",
            user_query="帮我搜索论文",
            max_iterations=20,
        )
    """

    def __init__(
        self,
        llm,                    # LLMMind 实例
        tool_registry,          # ToolRegistry 实例
        on_event: Callable | None = None,
        max_retries: int = 3,
    ):
        self.llm = llm
        self.tools = tool_registry
        self.on_event = on_event
        self.max_retries = max_retries

    def run(
        self,
        system_prompt: str,
        user_query: str,
        messages: list[dict] | None = None,
        max_iterations: int = 20,
        memory=None,            # MemoryManager, optional
        todos=None,             # TodoManager, optional
    ) -> str:
        """
        运行自主循环。

        返回: LLM 的最终文本回答
        """
        # 构建初始消息
        msgs: list[dict] = []

        # system prompt
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})

        # 已有消息（多轮对话）
        if messages:
            msgs.extend(messages)

        # 用户输入
        msgs.append({"role": "user", "content": user_query})

        tool_schemas = self.tools.to_openai_schemas() if self.tools.names() else None

        for iteration in range(max_iterations):
            # ── 调用 LLM ──────────────────────────
            kwargs = dict(
                model=self.llm.model_name,
                messages=msgs,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens,
            )
            if tool_schemas:
                kwargs["tools"] = tool_schemas
                kwargs["tool_choice"] = "auto"

            resp = self.llm._client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message

            # ── 情况1: 纯文本回答（无 tool_calls） ──
            if msg.content and not msg.tool_calls:
                return msg.content

            # ── 情况2: tool_calls ──────────────────
            if msg.tool_calls:
                # 记录 assistant 消息
                assistant_msg = {
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
                }
                msgs.append(assistant_msg)

                # 执行每个工具调用
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    # 发送 tool_start 事件
                    self._emit("tool_start", tool_name=tool_name, tool_args=tool_args)

                    # 执行
                    retries = 0
                    result = None
                    while retries <= self.max_retries:
                        try:
                            result = self.tools.call(tool_name, **tool_args)
                            break
                        except Exception as e:
                            retries += 1
                            if retries > self.max_retries:
                                result = f"[错误] {e}"
                                break
                            # 简单重试（延迟很短）
                            result = f"[重试 {retries}] {e}"

                    result_str = (
                        json.dumps(result, ensure_ascii=False)
                        if not isinstance(result, str) else str(result)
                    )

                    # 发送 tool_end 事件
                    success = not result_str.startswith("[错误]")
                    self._emit("tool_end", tool_name=tool_name,
                              content=result_str[:500], success=success)

                    # 回填 tool 结果
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

                    # 写入记忆
                    if memory:
                        memory.remember(
                            f"[{tool_name}] {result_str[:300]}",
                            kind="observation", importance=0.6,
                        )

                    # 更新 todo（如果工具名匹配某个 todo 项）
                    if todos:
                        for item in todos.list_by_status():
                            if tool_name in item.content:
                                todos.complete(item.id) if success else None

                # 继续循环，让 LLM 看到工具结果后决定下一步
                continue

            # ── 情况3: 空响应 → 继续 ────────────────
            continue

        # 超限: 强制 LLM 总结
        msgs.append({
            "role": "user",
            "content": "已达到最大迭代次数。请基于已有信息给出最终回答。",
        })
        resp = self.llm._client.chat.completions.create(
            model=self.llm.model_name,
            messages=msgs,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _emit(self, event_type: str, **kwargs):
        if self.on_event:
            from core.state import StreamEvent
            self.on_event(StreamEvent(type=event_type, **kwargs))


# ═══════════════════════════════════════════════════
# AgentEngine —— 固定流水线（保留，教育展示用）
# ═══════════════════════════════════════════════════

class AgentEngine:
    """
    轻量级状态机引擎 —— LangGraph 的替代品。

    用法:
        engine = AgentEngine(max_iterations=20)
        engine.add_node("planner", planner_node.plan)
        engine.add_node("executor", executor_node.execute)
        engine.add_node("reflector", reflector_node.reflect)
        engine.add_node("summarizer", summarizer_node.summarize)
        engine.set_entry_point("planner")
        engine.add_edge("planner", "executor")
        engine.add_edge("executor", "reflector")
        engine.add_edge("summarizer", END)
        engine.add_conditional_edges("reflector", router_fn, {
            "executor": "executor", "summarizer": "summarizer", "end": END,
        })
        for event in engine.stream(initial_state):
            print(f"[{event.node_name}]")
    """

    def __init__(self, max_iterations: int = 20, verbose: bool = True):
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, str] = {}
        self.conditional_edges: dict[str, tuple[RouterFn, dict[str, str]]] = {}
        self.entry_point: str | None = None
        self.max_iterations = max_iterations
        self.verbose = verbose

    def add_node(self, name: str, fn: NodeFn):
        self.nodes[name] = fn

    def set_entry_point(self, name: str):
        self.entry_point = name

    def add_edge(self, from_node: str, to_node: str):
        self.edges[from_node] = to_node

    def add_conditional_edges(self, source: str, router: RouterFn, route_map: dict[str, str]):
        self.conditional_edges[source] = (router, route_map)

    def _resolve_next(self, current_node: str, state: AgentState) -> str:
        if current_node in self.conditional_edges:
            router, route_map = self.conditional_edges[current_node]
            decision = router(state)
            return route_map.get(decision, END)
        if current_node in self.edges:
            return self.edges[current_node]
        return END

    def _apply_updates(self, state: AgentState, updates: dict):
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)

    def stream(self, initial_state: AgentState) -> Iterator[EngineEvent]:
        state = initial_state
        current_node = self.entry_point
        if not current_node:
            raise ValueError("entry_point not set")

        iteration = 0
        while current_node != END and iteration < self.max_iterations:
            iteration += 1
            state.iteration_count = iteration

            if current_node not in self.nodes:
                break

            updates = self.nodes[current_node](state)
            self._apply_updates(state, updates)

            replan_count = updates.get("replan_count", state.replan_count)
            if replan_count > state.max_replans:
                state.route_decision = RouteDecision.COMPLETE

            yield EngineEvent(
                node_name=current_node, updates=updates,
                state=state,
                route_decision=state.route_decision.value if state.route_decision else None,
            )

            next_node = self._resolve_next(current_node, state)
            if next_node == END:
                break
            current_node = next_node

        if iteration >= self.max_iterations:
            print(f"  [Engine] WARNING: max_iterations ({self.max_iterations}) reached")

        yield EngineEvent(node_name="__FINISH__", updates={}, state=state)

    def invoke(self, initial_state: AgentState) -> AgentState:
        final_state = initial_state
        for event in self.stream(initial_state):
            final_state = event.state
        return final_state
