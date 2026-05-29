"""
上下文窗口管理器 —— Agent 的"注意力系统"。

设计理念（教育目的）:
  LLM 的上下文窗口是稀缺资源（有 token 上限）。
  这个模块实现了 Gather → Select → Structure → Compress (GSSC) 流水线:

  ① Gather  — 从多源收集候选内容（系统提示、消息历史、记忆、工具结果）
  ② Select  — 按优先级和相关性筛选，确保不超预算
  ③ Structure— 组织成清晰的层级结构（Role → Task → State → Evidence → Context）
  ④ Compress — 超预算时压缩旧消息（截断 + 摘要）

Token 计数使用 tiktoken（精确），降级方案为字符数/4 估算。

用法:
    ctx = ContextManager(max_tokens=8000)
    messages = ctx.build(
        system_prompt="你是一个有用的助手",
        user_query="帮我搜索论文",
        history=[...],
        memories=["上次搜索返回了5篇论文"],
        todos="○ search_papers [pending]",
    )
    # messages 可以直接喂给 LLM
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import math


def _count_tokens(text: str) -> int:
    """估算 token 数。优先用 tiktoken，降级为字符/4。"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


@dataclass
class ContextPacket:
    """上下文信息包 —— GSSC 流水线中的基本单元"""
    content: str
    priority: int = 5          # 1~10, 数字越大越优先
    kind: str = "misc"         # system | task | memory | tool_result | history | todo
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = _count_tokens(self.content)


class ContextManager:
    """
    GSSC 上下文管理器。

    关键参数:
      max_tokens: 上下文窗口总 token 预算
      reserve_ratio: 保留给 LLM 输出的比例（默认 15%）
    """

    def __init__(self, max_tokens: int = 8000, reserve_ratio: float = 0.15):
        self.max_tokens = max_tokens
        self.reserve_ratio = reserve_ratio

    @property
    def _budget(self) -> int:
        return int(self.max_tokens * (1 - self.reserve_ratio))

    # ── 主入口 ────────────────────────────────────
    def build(
        self,
        system_prompt: str = "",
        user_query: str = "",
        history: list[dict] | None = None,
        memories: str = "",
        todos: str = "",
        tool_results: str = "",
    ) -> list[dict]:
        """
        构建最终的 LLM 消息列表。

        返回格式兼容 OpenAI ChatCompletion API。
        """
        packets: list[ContextPacket] = []

        # ① Gather —— 收集候选内容
        if system_prompt:
            packets.append(ContextPacket(
                content=system_prompt, priority=10, kind="system",
            ))
        if todos:
            packets.append(ContextPacket(
                content=todos, priority=8, kind="todo",
            ))
        if memories:
            packets.append(ContextPacket(
                content=memories, priority=7, kind="memory",
            ))
        if tool_results:
            packets.append(ContextPacket(
                content=tool_results, priority=6, kind="tool_result",
            ))
        if history:
            for msg in history[-20:]:  # 最多 20 条历史
                packets.append(ContextPacket(
                    content=msg.get("content", ""), priority=5, kind="history",
                ))

        # ② Select —— 按优先级筛选，不超预算
        selected = self._select(packets)

        # ③ Structure —— 组织为消息列表
        messages = self._structure(selected, user_query)

        # ④ Compress —— 超预算则压缩
        messages = self._compress(messages)

        return messages

    # ── ② Select ──────────────────────────────────
    def _select(self, packets: list[ContextPacket]) -> list[ContextPacket]:
        """按优先级排序，贪心填充到预算内"""
        packets.sort(key=lambda p: p.priority, reverse=True)

        selected: list[ContextPacket] = []
        used = 0

        for p in packets:
            if used + p.token_count <= self._budget:
                selected.append(p)
                used += p.token_count
            else:
                # 对低优先级内容截断
                remaining = self._budget - used
                if remaining > 100 and p.priority >= 6:
                    p.content = p.content[:remaining * 4]  # 粗略截断
                    p.token_count = _count_tokens(p.content)
                    selected.append(p)
                    used += p.token_count
                break  # 低优先级直接丢弃

        return selected

    # ── ③ Structure ───────────────────────────────
    def _structure(self, packets: list[ContextPacket], user_query: str) -> list[dict]:
        """组织为 OpenAI 消息格式"""
        messages: list[dict] = []

        # 系统级内容合并为一条 system 消息
        system_parts: list[str] = []
        other_parts: list[str] = []

        for p in packets:
            if p.kind in ("system", "todo", "memory", "tool_result"):
                system_parts.append(p.content)
            elif p.kind == "history":
                other_parts.append(p.content)

        # 构建 system prompt
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        # 历史消息（保留原始 role）
        for p in packets:
            if p.kind == "history":
                messages.append({"role": "user", "content": p.content})

        # 当前用户查询
        if user_query:
            messages.append({"role": "user", "content": user_query})

        return messages

    # ── ④ Compress ────────────────────────────────
    def _compress(self, messages: list[dict]) -> list[dict]:
        """超预算时压缩"""
        total = sum(_count_tokens(m.get("content", "")) for m in messages)
        if total <= self._budget:
            return messages

        # 保留 system + 最近 6 条，其余压缩为摘要
        system_msgs = [m for m in messages if m["role"] == "system"]
        other_msgs = [m for m in messages if m["role"] != "system"]

        if len(other_msgs) <= 6:
            return messages

        keep = other_msgs[-6:]
        old = other_msgs[:-6]

        summary = f"[已压缩 {len(old)} 条历史消息] " + "; ".join(
            m.get("content", "")[:80] for m in old[-3:]
        )

        return system_msgs + [
            {"role": "system", "content": summary}
        ] + keep

    # ── 工具: 计算 token 数 ───────────────────────
    @staticmethod
    def count_tokens(text: str) -> int:
        return _count_tokens(text)

    @staticmethod
    def count_messages(messages: list[dict]) -> int:
        return sum(_count_tokens(m.get("content", "")) for m in messages)
