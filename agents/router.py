"""
Router —— Reflector 之后的条件路由逻辑。

教育目的: 展示 agent 的"决策树"——根据自省结果决定下一步做什么。
这是 agent 控制流的核心。
"""

from core.state import AgentState, RouteDecision


def route_after_reflect(state: AgentState) -> str:
    """Reflector → 下一个节点"""

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
