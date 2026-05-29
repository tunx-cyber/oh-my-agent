"""
三层记忆系统 —— Agent 的"海马体"。

设计理念（教育目的）:
  真实的 Agent 需要记忆来避免重复犯错、关联历史信息。
  这里实现了人脑记忆的简化类比:

  ① 工作记忆 (Working Memory)
     - 当前上下文窗口内的消息，容量有限 (~7±2 条关键信息)
     - 类比: 你正在思考的事情

  ② 情节记忆 (Episodic Memory)
     - 本次会话的关键事件序列，按重要性过滤
     - 类比: 你今天经历了什么

  ③ 语义记忆 (Semantic Memory)
     - 长期向量存储，跨会话持久化
     - 类比: 你学到的知识和概念

工作流程:
  observe → remember(observation) → recall(query) → inject into context
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time
import hashlib
import json


@dataclass
class MemoryItem:
    """单条记忆"""
    content: str
    kind: str = "observation"   # observation | reflection | plan | lesson | fact
    importance: float = 0.5     # 0~1, 越高越容易留存
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "kind": self.kind,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def _hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()[:12]


class MemoryManager:
    """
    三层记忆管理器。

    用法:
        mem = MemoryManager(importance_threshold=0.6)
        mem.remember("搜索返回了3篇论文", kind="observation", importance=0.7)
        mem.remember("这一步失败了，因为API key无效", kind="reflection", importance=0.9)
        relevant = mem.recall("API key", k=5)
    """

    def __init__(self, importance_threshold: float = 0.6):
        self.working: list[MemoryItem] = []     # 工作记忆
        self.episodic: list[MemoryItem] = []    # 情节记忆
        self.semantic: list[MemoryItem] = []    # 语义记忆 (简化版，生产环境用向量数据库)
        self.threshold = importance_threshold

    # ── 写入 ──────────────────────────────────────
    def remember(
        self,
        content: str,
        kind: str = "observation",
        importance: float = 0.5,
        metadata: dict | None = None,
    ):
        """写入一条新记忆"""
        item = MemoryItem(
            content=content,
            kind=kind,
            importance=importance,
            metadata=metadata or {},
        )

        # 所有记忆先进工作区
        self.working.append(item)

        # 重要度超过阈值的进入情节记忆
        if importance >= self.threshold:
            self.episodic.append(item)

        # 高重要度的进入语义记忆（长期）
        if importance >= 0.8:
            self.semantic.append(item)

        # 工作记忆容量控制: 保留最近 20 条
        if len(self.working) > 20:
            self.working = self.working[-20:]

    # ── 检索 ──────────────────────────────────────
    def recall(self, query: str, k: int = 5, sources: str = "all") -> list[MemoryItem]:
        """
        检索与 query 最相关的记忆。

        sources: "working" | "episodic" | "semantic" | "all"

        当前使用关键词重叠 + 重要性加权排序。
        生产环境应替换为向量相似度检索（参考 memory/ 目录下的向量存储实现）。
        """
        candidates: list[MemoryItem] = []
        if sources in ("all", "working"):
            candidates.extend(self.working)
        if sources in ("all", "episodic"):
            candidates.extend(self.episodic)
        if sources in ("all", "semantic"):
            candidates.extend(self.semantic)

        query_words = set(query.lower().split())

        scored: list[tuple[float, MemoryItem]] = []
        for item in candidates:
            content_words = set(item.content.lower().split())
            # 关键词重叠率
            overlap = len(query_words & content_words) / max(len(query_words), 1)
            # 综合得分 = 关键词重叠 × 重要性
            score = overlap * (0.5 + 0.5 * item.importance)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 去重（按内容 hash）
        seen: set[str] = set()
        results: list[MemoryItem] = []
        for _, item in scored:
            h = item._hash()
            if h not in seen:
                seen.add(h)
                results.append(item)
            if len(results) >= k:
                break

        return results

    # ── 摘要（用于注入上下文） ─────────────────────
    def summarize_for_context(self, max_items: int = 5) -> str:
        """将重要记忆格式化为可注入 LLM 上下文的文本"""
        # 取情节记忆中重要性最高的几条
        top = sorted(self.episodic, key=lambda m: m.importance, reverse=True)[:max_items]
        if not top:
            return ""

        lines = ["[相关记忆]"]
        for m in top:
            lines.append(f"- [{m.kind}] {m.content}")
        return "\n".join(lines)

    # ── 工作记忆压缩 ──────────────────────────────
    def compress_working(self, keep_recent: int = 8) -> str | None:
        """
        当工作记忆过长时，将旧记忆压缩为一条摘要。
        返回摘要文本，或 None（无需压缩）。
        """
        if len(self.working) <= keep_recent:
            return None

        old = self.working[:-keep_recent]
        recent = self.working[-keep_recent:]

        # 简单拼接（生产环境用 LLM 生成摘要）
        summary_parts = []
        for m in old[-5:]:  # 最多摘录 5 条
            summary_parts.append(f"[{m.kind}] {m.content[:100]}")

        summary = f"[压缩记忆] 之前的 {len(old)} 条观察: " + "; ".join(summary_parts)
        self.working = [MemoryItem(content=summary, kind="summary", importance=0.8)] + recent

        return summary

    # ── 统计 ──────────────────────────────────────
    def stats(self) -> dict:
        return {
            "working_count": len(self.working),
            "episodic_count": len(self.episodic),
            "semantic_count": len(self.semantic),
            "avg_importance": (
                sum(m.importance for m in self.episodic) / max(len(self.episodic), 1)
            ),
        }

    def clear(self):
        """清空所有记忆"""
        self.working.clear()
        self.episodic.clear()
        self.semantic.clear()
