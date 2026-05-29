"""
Planner 节点 —— 将自然语言任务分解为可执行的原子步骤。

教育目的: 展示 LLM 如何通过结构化 Prompt 生成"计划"。
这是 agent 推理能力的核心: 不是一步到位，而是先规划再执行。
"""

from __future__ import annotations

from core.llm_mind import LLMMind
from core.state import AgentState, PlanStep, RouteDecision, StreamEvent, StreamCallback
from tools.base import ToolRegistry


PLAN_SYSTEM = """你是任务规划专家。将用户任务分解为可执行的原子步骤。

## 步骤类型
- **tool_call**: 调用工具（必须指定 tool_name 和 tool_args）
- **reason**: 纯推理/分析（不需要工具）
- **synthesize**: 汇总结果，产出最终答案（最后一步）

## 可用工具
{tools}

## 输出
```json
{{"steps": [{{"description": "...", "tool_name": "工具名或null", "tool_args": {{}}, "step_type": "tool_call|reason|synthesize"}}]}}
```
"""

REPLAN_SYSTEM = """你是任务重规划专家。当前计划执行遇到问题，请修正剩余步骤。
保留仍有效的步骤，替换或删除不可行的步骤。
如果任务已无法继续，用 synthesize 步骤基于已有结果给出最佳回答。

## 可用工具
{tools}

## 输出
```json
{{"steps": [...]}}
```
"""


class PlannerNode:

    def __init__(self, llm: LLMMind, registry: ToolRegistry):
        self.llm = llm
        self.registry = registry
        self.on_event: StreamCallback | None = None

    def _tools_desc(self) -> str:
        return self.registry.describe() or "(无)"

    def _parse(self, raw: dict) -> list[PlanStep]:
        return [
            PlanStep(
                description=s.get("description", ""),
                tool_name=s.get("tool_name"),
                tool_args=s.get("tool_args", {}),
                step_type=s.get("step_type", "tool_call"),
            )
            for s in raw.get("steps", [])
        ]

    def plan(self, state: AgentState) -> dict:
        msgs = [
            {"role": "system", "content": PLAN_SYSTEM.format(tools=self._tools_desc())},
            {"role": "user", "content": state.user_query},
        ]
        plan = self._parse(self.llm.chat_json(msgs))

        if not plan:
            plan = [PlanStep(description="分析并回答", step_type="synthesize")]

        # 发送计划事件
        self._emit(StreamEvent(
            type="plan", content="\n".join(
                f"  Step {i+1}: {s.description}" + (f" [{s.tool_name}]" if s.tool_name else "")
                for i, s in enumerate(plan)
            ),
            step_total=len(plan),
        ))

        return {
            "plan": plan,
            "current_step": 0,
            "route_decision": RouteDecision.CONTINUE,
        }

    def replan(self, state: AgentState) -> dict:
        plan = state.plan
        cur = state.current_step
        cur_step = state.current_plan_step()

        completed = "\n".join(
            f"  Step {i+1} [{s.status.value}]: {s.description} → {str(s.result)[:200]}"
            for i, s in enumerate(plan[:cur])
        ) or "  (无)"

        failed = ""
        if cur_step:
            failed = f"  失败: {cur_step.description} → {cur_step.error}"

        user = f"原始任务: {state.user_query}\n\n已完成:\n{completed}\n\n失败步骤:\n{failed}\n\n请输出修正后的剩余计划。"

        msgs = [
            {"role": "system", "content": REPLAN_SYSTEM.format(tools=self._tools_desc())},
            {"role": "user", "content": user},
        ]
        new_steps = self._parse(self.llm.chat_json(msgs))

        self._emit(StreamEvent(type="replan", content=f"修正为 {len(new_steps)} 个新步骤"))

        return {
            "plan": plan[:cur] + new_steps,
            "current_step": cur,
            "replan_count": state.replan_count + 1,
            "route_decision": RouteDecision.CONTINUE,
        }

    def _emit(self, event: StreamEvent):
        if self.on_event:
            self.on_event(event)
