"""
Executor 节点 —— 执行计划中的单个步骤。

教育目的: 展示 agent 如何将"计划"转化为"行动"。
三种执行模式:
  - tool_call:  调用外部工具（文件系统、代码执行、搜索等）
  - reason:     LLM 纯推理（分析、比较、归纳）
  - synthesize: 汇总所有步骤结果，生成最终答案

流式支持: 推理和综合阶段通过 stream_invoke 实时输出 LLM 文本。
"""

from __future__ import annotations
import json

from core.llm_mind import LLMMind
from core.state import AgentState, PlanStep, StepStatus, StreamEvent, StreamCallback
from tools.base import ToolRegistry


EXECUTOR_REASON_PROMPT = "你是分析助手。根据已有信息完成推理任务。简洁、准确。"

SYNTHESIZE_PROMPT = """你是回答生成器。根据任务描述和所有执行结果，生成面向用户的完整回答。
直接回答问题，结构清晰，引用具体数据，诚实说明失败步骤。"""


class ExecutorNode:

    def __init__(self, llm: LLMMind, registry: ToolRegistry):
        self.llm = llm
        self.registry = registry
        self.on_event: StreamCallback | None = None

    def execute(self, state: AgentState) -> dict:
        step = state.current_plan_step()
        if step is None:
            return self._synthesize(state)

        step.status = StepStatus.RUNNING

        if step.step_type == "reason":
            return self._reason(step, state)
        elif step.step_type == "synthesize":
            return self._synthesize(state)
        else:
            return self._tool_call(step, state)

    # ── 工具调用 ──────────────────────────────────
    def _tool_call(self, step: PlanStep, state: AgentState) -> dict:
        if not step.tool_name:
            return self._step_result(step, state, False, "[错误] 未指定工具名")

        # 发送工具调用开始事件
        self._emit(StreamEvent(
            type="tool_start", tool_name=step.tool_name,
            tool_args=step.tool_args, step_desc=step.description,
        ))

        try:
            result = self.registry.call(step.tool_name, **step.tool_args)
            result_str = (
                json.dumps(result, ensure_ascii=False)
                if not isinstance(result, str) else result
            )
            # 发送工具调用完成事件
            self._emit(StreamEvent(
                type="tool_end", tool_name=step.tool_name,
                content=result_str[:500], step_desc=step.description,
            ))
            return self._step_result(step, state, True, result_str)
        except Exception as e:
            self._emit(StreamEvent(type="error", content=str(e)))
            return self._step_result(step, state, False, str(e))

    # ── 纯推理（流式输出） ────────────────────────
    def _reason(self, step: PlanStep, state: AgentState) -> dict:
        context = "\n".join(
            f"Step {i+1}: {s.description} → {str(s.result)[:300]}"
            for i, s in enumerate(state.plan)
            if s.status == StepStatus.SUCCESS and s.result
        )

        messages = [
            {"role": "system", "content": EXECUTOR_REASON_PROMPT},
            {"role": "user", "content": (
                f"任务: {state.user_query}\n\n"
                f"已有信息:\n{context[-3000:]}\n\n"
                f"当前问题: {step.description}\n\n请分析推理。"
            )},
        ]

        # 流式输出推理过程
        chunks: list[str] = []
        for chunk in self.llm.stream_invoke(messages):
            chunks.append(chunk)
            self._emit(StreamEvent(type="reasoning", content=chunk))

        resp = "".join(chunks)
        self._emit(StreamEvent(type="reasoning_done"))
        return self._step_result(step, state, True, resp or "")

    # ── 最终综合（流式输出） ──────────────────────
    def _synthesize(self, state: AgentState) -> dict:
        results = "\n\n".join(
            f"### Step {i+1}: {s.description}\n[{s.status.value}] {str(s.result)[:500]}"
            for i, s in enumerate(state.plan) if s.result
        )
        reflections = "\n".join(
            f"- [{r.decision.value}] {r.critique}" for r in state.reflections[-3:]
        )

        messages = [
            {"role": "system", "content": SYNTHESIZE_PROMPT},
            {"role": "user", "content": (
                f"## 任务\n{state.user_query}\n\n"
                f"## 执行结果\n{results}\n\n"
                f"## 自省\n{reflections}\n\n请生成最终回答。"
            )},
        ]

        # 流式输出最终回答
        chunks: list[str] = []
        for chunk in self.llm.stream_invoke(messages):
            chunks.append(chunk)
            self._emit(StreamEvent(type="text", content=chunk))

        resp = "".join(chunks)
        self._emit(StreamEvent(type="text_done"))
        return {"final_answer": resp or "", "is_complete": True}

    # ── 辅助 ──────────────────────────────────────
    def _step_result(self, step: PlanStep, state: AgentState, success: bool, result: str) -> dict:
        step.status = StepStatus.FAILED if not success else StepStatus.SUCCESS
        step.result = result
        if not success:
            step.error = result
            step.retry_count += 1

        return {
            "plan": state.plan,
            "step_results": [{
                "step_id": step.id, "step_desc": step.description,
                "result": result, "status": step.status.value,
            }],
        }

    def _emit(self, event: StreamEvent):
        if self.on_event:
            self.on_event(event)
