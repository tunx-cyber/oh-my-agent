"""
Reflector 节点 —— 质量门控，每步执行后评估结果。

教育目的: 展示 agent 如何"自省"——不是盲目执行，而是每步检查质量。
这是提升 agent 可靠性的关键设计。

决策逻辑:
  continue  → 进入下一步
  revise    → 重新执行当前步骤（修正错误）
  replan    → 重新规划剩余步骤
  complete  → 任务完成，进入总结
  escalate  → 超出能力，需要人工

置信度门控: confidence < 阈值 的 continue 自动升级为 revise
"""

from __future__ import annotations

from core.llm_mind import LLMMind
from core.state import AgentState, StepStatus, RouteDecision, ReflectionResult, StreamEvent, StreamCallback


REFLECT_PROMPT = """你是质量审查专家。评估 Agent 最近一步的结果，决定下一步行动。

## 决策选项
- continue: 结果合格，继续下一步
- revise: 结果有问题，修正当前步骤（给 fix_suggest）
- replan: 路径不通，重规划剩余步骤
- complete: 信息足够，产出最终答案

## 输出格式
```json
{"decision":"continue|revise|replan|complete","confidence":0.8,"critique":"评估分析","fix_suggest":"修正建议或null"}
```
"""


class ReflectorNode:

    def __init__(self, llm: LLMMind, confidence_gate: float = 0.6):
        self.llm = llm
        self.confidence_gate = confidence_gate
        self.on_event: StreamCallback | None = None

    def reflect(self, state: AgentState) -> dict:
        step = state.current_plan_step()
        if step is None:
            return {"route_decision": RouteDecision.COMPLETE}

        # 构建执行历史
        history = "\n".join(
            f"{'>>>' if i == state.current_step else '   '} Step {i+1} [{s.status.value}]: {s.description}\n"
            f"        Result: {str(s.result)[:300]}"
            for i, s in enumerate(state.plan[:state.current_step + 1])
        )

        prompt = (
            f"## 任务\n{state.user_query}\n\n"
            f"## 执行记录\n{history}\n\n"
            f"## 当前步骤\n{step.description}\n"
            f"工具: {step.tool_name or 'N/A'}\n"
            f"结果: {str(step.result)[:500]}\n"
            f"状态: {step.status.value}\n"
            f"重试: {step.retry_count}/{state.max_retries}\n\n"
            "请评估。"
        )

        raw = self.llm.chat_json([
            {"role": "system", "content": REFLECT_PROMPT},
            {"role": "user", "content": prompt},
        ])

        decision_str = raw.get("decision", "continue")
        confidence = float(raw.get("confidence", 0.5))
        critique = raw.get("critique", "")
        fix_suggest = raw.get("fix_suggest")

        # 置信度门控
        if decision_str == "continue" and confidence < self.confidence_gate:
            decision_str = "revise"
            critique += f" [低置信度({confidence:.2f})自动升级]"

        # 失败超限 → replan
        if (step.status == StepStatus.FAILED
                and step.retry_count >= state.max_retries
                and decision_str == "revise"):
            decision_str = "replan"
            critique += " [超最大重试，升级为 replan]"

        decision = RouteDecision(decision_str)

        # 发送自省事件
        self._emit(StreamEvent(
            type="reflect",
            content=critique,
            confidence=confidence,
            critique=decision_str,
        ))

        return {
            "route_decision": decision,
            "reflections": [ReflectionResult(
                decision=decision, confidence=confidence,
                critique=critique, fix_suggest=fix_suggest,
            )],
        }

    def _emit(self, event: StreamEvent):
        if self.on_event:
            self.on_event(event)
