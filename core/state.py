"""
Agent 状态定义 —— 贯穿整个 agent 循环的共享数据结构。

设计理念（教育目的）:
  所有节点通过共享状态通信。每个节点读取状态 → 决策 → 返回部分更新。
  这模拟了大脑的工作方式: 不同脑区共享同一份"工作记忆"。

不依赖 LangGraph，纯 dataclass 实现。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import time
import uuid


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    REVISED = "revised"


class RouteDecision(str, Enum):
    CONTINUE = "continue"
    REVISE = "revise"
    REPLAN = "replan"
    COMPLETE = "complete"


@dataclass
class PlanStep:
    """计划中的单个步骤"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    step_type: str = "tool_call"   # tool_call | reason | synthesize
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str | None = None
    retry_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "description": self.description,
            "tool_name": self.tool_name, "tool_args": self.tool_args,
            "step_type": self.step_type, "status": self.status.value,
            "result": self.result, "error": self.error, "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlanStep":
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            description=d.get("description", ""),
            tool_name=d.get("tool_name"),
            tool_args=d.get("tool_args", {}),
            step_type=d.get("step_type", "tool_call"),
            status=StepStatus(d.get("status", "pending")),
            result=d.get("result"), error=d.get("error"),
            retry_count=d.get("retry_count", 0),
        )


@dataclass
class ReflectionResult:
    """自省结果"""
    decision: RouteDecision = RouteDecision.CONTINUE
    confidence: float = 0.0
    critique: str = ""
    fix_suggest: str | None = None

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value, "confidence": self.confidence,
            "critique": self.critique, "fix_suggest": self.fix_suggest,
        }


# ── 流式事件 ──────────────────────────────────────
@dataclass
class StreamEvent:
    """Agent 运行时向外发送的事件，用于实时展示进度"""
    type: str = ""          # start | plan | tool_start | tool_end
                            # | reasoning | reasoning_done | text | text_done
                            # | reflect | replan | dispatch | complete | error
    content: str = ""       # 事件携带的文本
    step_index: int = 0
    step_total: int = 0
    step_desc: str = ""
    tool_name: str = ""
    tool_args: dict | None = None
    confidence: float = 0.0
    critique: str = ""
    success: bool = True    # tool_end 时表示工具是否成功


# callback 类型: 节点通过它向外发送 StreamEvent
StreamCallback = Callable[[StreamEvent], None]


@dataclass
class MemoryEntry:
    """记忆条目"""
    content: str = ""
    kind: str = "observation"
    importance: float = 0.5
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"content": self.content, "kind": self.kind, "importance": self.importance}


@dataclass
class AgentState:
    """
    贯穿整个 agent 循环的全局状态。

    每个节点执行完成后返回 dict，engine 自动合并到 state 的对应字段。
    """

    # ── 对话 ──
    messages: list[dict] = field(default_factory=list)
    user_query: str = ""

    # ── 规划 ──
    plan: list[PlanStep] = field(default_factory=list)
    current_step: int = 0

    # ── 执行记录 ──
    step_results: list[dict] = field(default_factory=list)

    # ── 自省 ──
    reflections: list[ReflectionResult] = field(default_factory=list)
    route_decision: RouteDecision = RouteDecision.CONTINUE

    # ── 控制流 ──
    retry_count: int = 0
    replan_count: int = 0
    max_replans: int = 2
    max_retries: int = 3
    max_iterations: int = 20
    iteration_count: int = 0
    is_complete: bool = False

    # ── 最终输出 ──
    final_answer: str = ""

    # ── 辅助方法 ──
    def current_plan_step(self) -> PlanStep | None:
        if 0 <= self.current_step < len(self.plan):
            return self.plan[self.current_step]
        return None

    def advance_step(self):
        self.current_step += 1

    def is_plan_exhausted(self) -> bool:
        return self.current_step >= len(self.plan)

    def to_dict(self) -> dict:
        return {
            "user_query": self.user_query,
            "plan": [s.to_dict() for s in self.plan],
            "current_step": self.current_step,
            "step_results": self.step_results,
            "reflections": [r.to_dict() for r in self.reflections],
            "route_decision": self.route_decision.value,
            "retry_count": self.retry_count,
            "replan_count": self.replan_count,
            "is_complete": self.is_complete,
            "final_answer": self.final_answer,
        }
