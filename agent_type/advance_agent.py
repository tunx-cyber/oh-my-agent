"""
DeepAgent —— 规划 → 执行 → 自省 → 重规划 完整闭环。

组件:
  PlannerNode   — 将用户任务分解为原子步骤序列
  ExecutorNode  — 执行单个步骤 (工具调用 / 推理 / 综合)
  ReflectorNode — 每步后质量评估，决定 continue/revise/replan/complete
  SummarizerNode— 生成最终面向用户的回答

不依赖 LangGraph，使用自己的 AgentEngine 编排。
"""

from __future__ import annotations
import json

from core.llm_mind import LLMMind
from core.state import (
    AgentState, PlanStep, StepStatus,
    RouteDecision, ReflectionResult, MemoryEntry,
)
from core.agent_loop import AgentEngine, END
from tools.base import ToolRegistry


# ═══════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════

PLANNER_SYSTEM = """你是一个精确的任务规划专家。

你的职责是将用户的复杂任务分解为一系列 **可执行的原子步骤**。

## 步骤类型
- **tool_call**: 调用一个具体工具（必须指定 tool_name 和 tool_args）
- **reason**: 纯推理/分析/整理信息（tool_name 为 null）
- **synthesize**: 汇总所有已有结果，产出最终答案（最后一步）

## 可用工具
{tool_descriptions}

## 输出格式
严格以 JSON 对象返回:
```json
{{
  "steps": [
    {{"description": "...", "tool_name": "工具名或null", "tool_args": {{}}, "step_type": "tool_call|reason|synthesize"}}
  ]
}}
```
"""

REPLANNER_SYSTEM = """你是任务重规划专家。执行中遇到了问题，需要修正剩余计划。
保留仍有效的步骤，修改或删除因失败而不可行的步骤。
如任务无法继续，输出 synthesize 步骤基于已有结果给出最佳回答。

## 可用工具
{tool_descriptions}

## 输出格式
严格以 JSON 对象返回:
```json
{{"steps": [...]}}
```
"""

EXECUTOR_REASON_SYSTEM = """你是分析推理助手。根据已有执行记录完成当前推理步骤。简洁、准确。"""

SYNTHESIZER_SYSTEM = """你是最终回答生成器。根据任务描述和所有执行结果，产出完整、面向用户的最终回答。"""

REFLECTOR_SYSTEM = """你是严格的质量审查专家。评估 Agent 最近一步的执行结果并决定后续行动。

## 决策选项
- **continue**: 结果合格，继续下一步
- **revise**: 结果有问题，修正当前步骤（给出具体修正建议）
- **replan**: 当前路径走不通，需要重新规划后续步骤
- **complete**: 所有必要信息已收集，产出最终答案
- **escalate**: 任务超出能力范围，需要人工介入

## 输出格式
```json
{{"decision":"continue|revise|replan|complete|escalate","confidence":0.8,"critique":"...","fix_suggest":"..."}}
```
"""


# ═══════════════════════════════════════════════════
# Planner 节点
# ═══════════════════════════════════════════════════

class PlannerNode:
    """规划节点 —— 生成/修正执行计划"""

    def __init__(self, llm: LLMMind, registry: ToolRegistry):
        self.llm = llm
        self.registry = registry

    def _tool_descriptions(self) -> str:
        lines = []
        for t in self.registry.tools.values():
            schema = t.to_openai_schema()
            f = schema["function"]
            params = json.dumps(f.get("parameters", {}), ensure_ascii=False, indent=2)
            lines.append(f"### {f['name']}\n{f['description']}\n```{params}```")
        return "\n\n".join(lines) if lines else "(无可用工具)"

    def _parse_plan(self, raw: dict) -> list[PlanStep]:
        steps = raw.get("steps", [])
        return [
            PlanStep(
                description=s.get("description", ""),
                tool_name=s.get("tool_name"),
                tool_args=s.get("tool_args", {}),
                step_type=s.get("step_type", "tool_call"),
            )
            for s in steps
        ]

    def plan(self, state: AgentState) -> dict:
        """初始规划"""
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM.format(
                tool_descriptions=self._tool_descriptions()
            )},
            {"role": "user", "content": state.user_query},
        ]
        raw = self.llm.chat_json(messages)
        plan = self._parse_plan(raw)

        return {
            "plan": plan,
            "current_step": 0,
            "route_decision": RouteDecision.CONTINUE,
            "memory_entries": [MemoryEntry(
                content=f"初始计划: {' → '.join(s.description for s in plan)}",
                kind="plan", importance=0.8,
            )],
        }

    def replan(self, state: AgentState) -> dict:
        """重规划 —— 保留已完成步骤，修正剩余部分"""
        plan = state.plan
        cur_idx = state.current_step
        cur_step = state.current_plan_step()

        completed = "\n".join(
            f"  Step {i+1} [{s.status.value}]: {s.description} → {str(s.result)[:200]}"
            for i, s in enumerate(plan[:cur_idx])
        ) or "  (无)"

        failed = ""
        if cur_step:
            failed = (f"  Step {cur_idx+1} [{cur_step.status.value}]: "
                      f"{cur_step.description} → Error: {cur_step.error}")

        user_msg = (
            f"原始任务: {state.user_query}\n\n"
            f"已完成:\n{completed}\n\n"
            f"失败步骤:\n{failed}\n\n"
            f"请输出修正后的剩余计划。"
        )

        messages = [
            {"role": "system", "content": REPLANNER_SYSTEM.format(
                tool_descriptions=self._tool_descriptions()
            )},
            {"role": "user", "content": user_msg},
        ]
        raw = self.llm.chat_json(messages)
        new_steps = self._parse_plan(raw)

        # 已完成部分 + 新计划
        updated_plan = plan[:cur_idx] + new_steps

        return {
            "plan": updated_plan,
            "current_step": cur_idx,
            "replan_count": state.replan_count + 1,
            "route_decision": RouteDecision.CONTINUE,
            "memory_entries": [MemoryEntry(
                content=f"重规划: 从 step {cur_idx+1} 起替换为 {len(new_steps)} 个新步骤",
                kind="plan", importance=0.9,
            )],
        }



# ═══════════════════════════════════════════════════
# Executor 节点
# ═══════════════════════════════════════════════════

class ExecutorNode:
    """执行节点 —— 按 step_type 分流执行"""

    def __init__(self, llm: LLMMind, registry: ToolRegistry):
        self.llm = llm
        self.registry = registry

    def execute(self, state: AgentState) -> dict:
        step = state.current_plan_step()
        if step is None:
            return self._synthesize(state)

        step.status = StepStatus.RUNNING

        step_type = step.step_type
        if step_type == "reason":
            return self._reason(step, state)
        elif step_type == "synthesize":
            return self._synthesize(state)
        else:
            return self._tool_call(step, state)

    def _tool_call(self, step: PlanStep, state: AgentState) -> dict:
        if not step.tool_name:
            return self._result(step, state, success=False, result="[Error] 未指定工具")

        try:
            observation = self.registry.call(step.tool_name, step.tool_args)
            result_str = json.dumps(observation, ensure_ascii=False) if not isinstance(observation, str) else observation
            return self._result(step, state, success=True, result=result_str)
        except Exception as e:
            return self._result(step, state, success=False, result=str(e))

    def _reason(self, step: PlanStep, state: AgentState) -> dict:
        context_parts = []
        for i, s in enumerate(state.plan):
            if s.status == StepStatus.SUCCESS and s.result:
                context_parts.append(f"Step {i+1}: {s.description} → {str(s.result)[:300]}")

        prompt = (
            f"任务: {state.user_query}\n\n"
            f"当前步骤: {step.description}\n\n"
            f"已有观察:\n" + "\n".join(context_parts[-6:]) + "\n\n"
            "请完成分析推理。"
        )

        resp = self.llm.invoke([
            {"role": "system", "content": EXECUTOR_REASON_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        return self._result(step, state, success=True, result=resp)

    def _synthesize(self, state: AgentState) -> dict:
        results_parts = []
        for i, s in enumerate(state.plan):
            if s.result:
                results_parts.append(
                    f"### Step {i+1}: {s.description}\n"
                    f"Status: {s.status.value}\nResult: {str(s.result)[:500]}"
                )

        reflections = "\n".join(
            f"- [{r.decision.value}] {r.critique}"
            for r in state.reflections[-3:]
        )

        prompt = (
            f"## 用户任务\n{state.user_query}\n\n"
            f"## 执行结果\n" + "\n\n".join(results_parts) + "\n\n"
            f"## 自省记录\n{reflections}\n\n"
            "请给出完整、准确的最终回答。"
        )

        resp = self.llm.invoke([
            {"role": "system", "content": SYNTHESIZER_SYSTEM},
            {"role": "user", "content": prompt},
        ])

        return {
            "final_answer": resp,
            "is_complete": True,
            "step_results": [{"step_id": "synthesize", "result": resp, "status": "success"}],
            "memory_entries": [MemoryEntry(
                content=f"任务完成: {state.user_query[:100]}", kind="lesson", importance=0.85,
            )],
        }

    def _result(self, step: PlanStep, state: AgentState, *, success: bool, result: str) -> dict:
        step.status = StepStatus.FAILED if not success else StepStatus.SUCCESS
        step.result = result
        if not success:
            step.error = result
            step.retry_count += 1

        return {
            "plan": state.plan,
            "step_results": [{
                "step_id": step.id,
                "step_desc": step.description,
                "result": result,
                "status": step.status.value,
            }],
            "retry_count": step.retry_count,
        }


# ═══════════════════════════════════════════════════
# Reflector 节点
# ═══════════════════════════════════════════════════

class ReflectorNode:
    """自省节点 —— 质量门控核心"""

    def __init__(self, llm: LLMMind, confidence_gate: float = 0.6):
        self.llm = llm
        self.confidence_gate = confidence_gate

    def reflect(self, state: AgentState) -> dict:
        plan = state.plan
        cur_idx = state.current_step
        current_step = state.current_plan_step()

        if current_step is None:
            return {"route_decision": RouteDecision.COMPLETE}

        # 构建执行历史
        history_lines = []
        for i, s in enumerate(plan[:cur_idx + 1]):
            marker = " >>>" if i == cur_idx else "    "
            history_lines.append(
                f"{marker} Step {i+1} [{s.status.value}]: {s.description}\n"
                f"        Result: {str(s.result)[:300]}"
            )

        prompt = (
            f"## 任务\n{state.user_query}\n\n"
            f"## 执行记录\n" + "\n".join(history_lines) + "\n\n"
            f"## 当前步骤\n描述: {current_step.description}\n"
            f"工具: {current_step.tool_name or 'N/A'}\n"
            f"结果: {str(current_step.result)[:500]}\n"
            f"状态: {current_step.status.value}\n"
            f"重试: {current_step.retry_count}/{state.max_retries}\n\n"
            "请评估并决定后续行动。"
        )

        raw = self.llm.chat_json([
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {"role": "user", "content": prompt},
        ])

        decision_str = raw.get("decision", "continue")
        confidence = float(raw.get("confidence", 0.5))
        critique = raw.get("critique", "")
        fix_suggest = raw.get("fix_suggest")

        # 置信度门控: 低置信度的 continue 自动升级为 revise
        if decision_str == "continue" and confidence < self.confidence_gate:
            decision_str = "revise"
            critique += f" [低置信度({confidence})自动升级为 REVISE]"

        # 失败且超最大重试 → 强制 replan
        if (current_step.status == StepStatus.FAILED
                and current_step.retry_count >= state.max_retries
                and decision_str == "revise"):
            decision_str = "replan"
            critique += " [超最大重试次数，自动升级为 REPLAN]"

        decision = RouteDecision(decision_str)
        reflection = ReflectionResult(
            decision=decision,
            confidence=confidence,
            critique=critique,
            fix_suggest=fix_suggest,
        )

        return {
            "route_decision": decision,
            "reflections": [reflection],
            "memory_entries": [MemoryEntry(
                content=f"[Reflection] {critique} → {decision.value} ({confidence:.2f})",
                kind="reflection", importance=confidence,
            )],
        }



# ═══════════════════════════════════════════════════
# Summarizer 节点
# ═══════════════════════════════════════════════════

class SummarizerNode:
    """总结节点 —— 生成最终回答"""

    SUMMARIZE_SYSTEM = """你是专业的回答生成器。根据全部执行结果生成高质量最终回答。
要求: 直接回答用户问题、引用具体数据、结构清晰、诚实说明失败步骤。"""

    def __init__(self, llm: LLMMind):
        self.llm = llm

    def summarize(self, state: AgentState) -> dict:
        if state.final_answer and state.is_complete:
            return {}

        results_parts = []
        for i, s in enumerate(state.plan):
            icon = "✓" if s.status == StepStatus.SUCCESS else "✗"
            results_parts.append(
                f"{icon} Step {i+1}: {s.description}\n  结果: {str(s.result)[:400]}"
            )

        reflections = "\n".join(
            f"- [{r.decision.value}] {r.critique}"
            for r in state.reflections[-5:]
        )

        prompt = (
            f"## 用户任务\n{state.user_query}\n\n"
            f"## 执行步骤及结果\n" + "\n\n".join(results_parts) + "\n\n"
            f"## 自省记录\n{reflections}\n\n"
            "请生成最终回答。"
        )

        resp = self.llm.invoke([
            {"role": "system", "content": self.SUMMARIZE_SYSTEM},
            {"role": "user", "content": prompt},
        ])

        return {
            "final_answer": resp,
            "is_complete": True,
            "memory_entries": [MemoryEntry(
                content=f"任务完成: {state.user_query[:100]}", kind="lesson", importance=0.85,
            )],
        }


# ═══════════════════════════════════════════════════
# 路由函数
# ═══════════════════════════════════════════════════

def route_after_reflect(state: AgentState) -> str:
    """Reflector 之后的条件路由"""
    decision = state.route_decision

    if decision == RouteDecision.COMPLETE:
        return "summarizer"
    elif decision == RouteDecision.CONTINUE:
        state.advance_step()
        if state.is_plan_exhausted():
            return "summarizer"
        return "executor"
    elif decision == RouteDecision.REVISE:
        # 不推进索引，重新执行当前步骤
        return "executor"
    elif decision == RouteDecision.REPLAN:
        return "planner_replan"
    elif decision == RouteDecision.ESCALATE:
        return "end"
    return "executor"


# ═══════════════════════════════════════════════════
# DeepAgent —— 顶层封装
# ═══════════════════════════════════════════════════

class DeepAgent:
    """完整的 DeepAgent —— 规划→执行→自省→总结 闭环"""

    def __init__(
        self,
        name: str,
        llm: LLMMind,
        tools: list | None = None,
        max_retries: int = 3,
        max_replans: int = 2,
        max_iterations: int = 20,
        confidence_gate: float = 0.6,
        verbose: bool = True,
    ):
        self.name = name
        self.llm = llm
        self.verbose = verbose

        # 工具注册
        self.tool_registry = ToolRegistry()
        if tools:
            for tool in tools:
                self.tool_registry.register_tool(tool)

        # 各节点
        self.planner = PlannerNode(llm, self.tool_registry)
        self.executor = ExecutorNode(llm, self.tool_registry)
        self.reflector = ReflectorNode(llm, confidence_gate=confidence_gate)
        self.summarizer = SummarizerNode(llm)

        # 引擎配置
        self.max_retries = max_retries
        self.max_replans = max_replans
        self.max_iterations = max_iterations

        if verbose:
            print(f"DeepAgent [{name}] 初始化完成 "
                  f"(max_retries={max_retries}, max_replans={max_replans})")

    def run(self, input_text: str, **kwargs) -> str:
        """运行 agent，返回最终答案"""
        # 构建引擎
        engine = AgentEngine(max_iterations=self.max_iterations, verbose=self.verbose)

        engine.add_node("planner", self.planner.plan)
        engine.add_node("planner_replan", self.planner.replan)
        engine.add_node("executor", self.executor.execute)
        engine.add_node("reflector", self.reflector.reflect)
        engine.add_node("summarizer", self.summarizer.summarize)

        engine.set_entry_point("planner")

        engine.add_edge("planner", "executor")
        engine.add_edge("planner_replan", "executor")
        engine.add_edge("executor", "reflector")
        engine.add_edge("summarizer", END)

        engine.add_conditional_edges("reflector", route_after_reflect, {
            "executor": "executor",
            "planner_replan": "planner_replan",
            "summarizer": "summarizer",
            "end": END,
        })

        # 初始状态
        state = AgentState(
            messages=[{"role": "user", "content": input_text}],
            user_query=input_text,
            max_retries=self.max_retries,
            max_replans=self.max_replans,
            max_iterations=self.max_iterations,
        )

        # 执行
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  DeepAgent [{self.name}] 开始处理: {input_text[:60]}...")
            print(f"{'='*70}")

        final_state = engine.invoke(state)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  DeepAgent [{self.name}] 完成")
            print(f"{'='*70}")
            print(f"\n最终答案:\n{final_state.final_answer}")

        return final_state.final_answer
