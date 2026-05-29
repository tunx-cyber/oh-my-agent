"""
OrchestratorAgent —— 多 Agent 编排器。

设计理念（教育目的）:
  这是多 Agent 协作的"指挥家"。核心思路:
  1. Orchestrator 本身是一个 DeepAgent（拥有规划→执行→自省能力）
  2. 额外配备"人事管理"工具: dispatch / query / wait / list
  3. 子 Agent 在后台线程中独立运行，状态通过 AgentPool 同步
  4. 用户可随时询问子Agent进度，Orchestrator查询池状态并回答

工作流程:
  用户: "帮我研究 AI 芯片市场并写报告"
    Orchestrator 规划:
      Step 1: dispatch_agent("searcher", "搜索AI芯片市场数据")
      Step 2: dispatch_agent("analyst", "分析趋势")
      Step 3: wait_all → 汇总 → 生成最终报告
    (searcher 和 analyst 并行运行)

  用户中途: "searcher 怎么样了?"
    Orchestrator 查询: query_agent("searcher") → "已完成，找到5份报告"
    回答用户: "searcher 已完成，正在等待 analyst..."
"""

from __future__ import annotations
from typing import Callable

from core.llm_mind import LLMMind
from core.state import StreamEvent
from core.agent_pool import AgentPool
from tools.base import Tool, ToolRegistry, tool
from agents.deep_agent import DeepAgent


# ═══════════════════════════════════════════════════
# 编排器工具 — 将 AgentPool 能力暴露为 LLM 可调用的工具
# ═══════════════════════════════════════════════════

def _make_orchestrator_tools(
    pool: AgentPool,
    shared_tools: list,
    llm: LLMMind,
    on_event: Callable | None,
) -> list[Tool]:
    """
    构造编排器专用工具。

    用普通函数 + Tool 构造器（不用 @tool 装饰器），
    因为 dispatch_agent 需要闭包捕获 pool/llm/on_event。
    """

    # ── dispatch_agent ─────────────────────────────
    def _dispatch_agent(agent_name: str, task: str, tools_hint: str = "read_file, python_exec") -> str:
        """派发任务给子Agent（后台异步运行）"""
        # 按 tools_hint 筛选工具
        hints = set(tools_hint.replace(",", " ").split())
        sub_tools = [t for t in shared_tools if t.name in hints]
        if not sub_tools:
            sub_tools = shared_tools

        if on_event:
            on_event(StreamEvent(
                type="dispatch", tool_name=agent_name,
                content=f"→ [{agent_name}]: {task[:100]}",
            ))

        def _factory():
            return DeepAgent(
                name=agent_name, llm=llm, tools=sub_tools,
                max_iterations=10,
            )

        try:
            pool.spawn(name=agent_name, agent_factory=_factory, task=task)
            return f"✅ 已派发 [{agent_name}]: {task[:120]}"
        except RuntimeError as e:
            return f"❌ 派发失败: {e}"

    # ── query_agent ────────────────────────────────
    def _query_agent(agent_name: str) -> str:
        """查询子Agent状态（不阻塞）"""
        h = pool.get_handle(agent_name)
        if h is None:
            return f"[{agent_name}] 不存在。可用: {list(pool._agents.keys())}"
        return h.summary()

    # ── list_agents ────────────────────────────────
    def _list_agents() -> str:
        """列出所有子Agent"""
        return pool.list_all()

    # ── wait_agent ─────────────────────────────────
    def _wait_agent(agent_name: str, timeout: int = 60) -> str:
        """等待子Agent完成并获取结果（阻塞）"""
        return pool.result(agent_name, timeout=timeout)

    # ── wait_all_agents ───────────────────────────
    def _wait_all_agents(timeout: int = 120) -> str:
        """等待所有子Agent完成"""
        results = pool.wait_all(timeout=timeout)
        lines = ["📋 所有子Agent结果:"]
        for name, result in results.items():
            lines.append(f"\n### [{name}]\n{result[:500]}")
        return "\n".join(lines)

    return [
        Tool(func=_dispatch_agent, name="dispatch_agent",
             description="派发任务给子Agent（后台异步运行）。子Agent会独立完成规划→执行→自省全过程。"
                         "参数: agent_name(子Agent名), task(任务描述), tools_hint(建议工具,逗号分隔)"),
        Tool(func=_query_agent, name="query_agent",
             description="查询指定子Agent的当前状态和进度（不阻塞）。参数: agent_name"),
        Tool(func=_list_agents, name="list_agents",
             description="列出所有子Agent及其当前状态"),
        Tool(func=_wait_agent, name="wait_agent",
             description="等待指定子Agent完成并获取最终结果（会阻塞）。参数: agent_name, timeout(默认60秒)"),
        Tool(func=_wait_all_agents, name="wait_all_agents",
             description="等待所有子Agent完成并返回汇总结果。参数: timeout(默认120秒)"),
    ]


# ═══════════════════════════════════════════════════
# OrchestratorAgent
# ═══════════════════════════════════════════════════

class OrchestratorAgent:
    """
    多 Agent 编排器 —— 协调多个子 Agent 完成复杂任务。

    用法:
        orch = OrchestratorAgent(llm=llm, shared_tools=[search, python_exec])

        # 一次性任务
        result = orch.run("搜索AI市场数据，分析趋势，写报告")

        # 交互式对话（保持AgentPool状态跨轮次）
        orch.chat("派searcher去搜索AI芯片市场")
        orch.chat("派analyst去分析趋势")
        orch.chat("searcher完成了吗？")
        orch.chat("analyst怎么样？")
        orch.chat("汇总所有结果")
    """

    ORCHESTRATOR_PROMPT = """你是多 Agent 编排器（Orchestrator），负责协调多个子 Agent 完成复杂任务。

## 核心能力
你拥有 dispatch_agent / query_agent / list_agents / wait_agent / wait_all_agents 工具。

## 工作原则
1. **分解任务**: 将复杂任务拆为子任务，派给不同子Agent
2. **并行执行**: 独立子任务同时派发（子Agent并行运行）
3. **主动查询**: 派发后用 list_agents 确认启动；等一会用 query_agent 检查进度
4. **响应询问**: 用户问进度时立即查询，不猜测
5. **汇总整合**: 所有子Agent完成后用 wait_all_agents 收集结果，综合回答

## 子Agent命名建议
- searcher: 搜索/查找
- analyst: 分析/推理
- coder: 编写代码
- writer: 撰写文档

## 重要提醒
- 派发任务描述必须具体（含预期输出格式）
- 不要创建过多子Agent（2-3个为宜）
- 用户问进度时立刻查询，不要说"我猜..."
"""

    def __init__(
        self,
        llm: LLMMind,
        shared_tools: list | None = None,
        system_prompt: str = "",
        on_event: Callable | None = None,
    ):
        self.llm = llm
        self.on_event = on_event
        self.pool = AgentPool()
        self.shared_tools = shared_tools or []

        # 构造编排工具
        orch_tools = _make_orchestrator_tools(
            self.pool, self.shared_tools, llm, on_event,
        )

        # 合并: 编排工具 + 普通工具（去重）
        orch_names = {t.name for t in orch_tools}
        all_tools = orch_tools + [
            t for t in self.shared_tools if t.name not in orch_names
        ]

        # 编排器本身是一个 DeepAgent
        self.agent = DeepAgent(
            name="orchestrator",
            llm=llm,
            tools=all_tools,
            system_prompt=system_prompt or self.ORCHESTRATOR_PROMPT,
            max_iterations=30,
        )

    # ── 主入口 ────────────────────────────────────
    def run(self, task: str, stream: bool = True) -> str:
        """一次性运行编排任务"""
        if self.on_event:
            self.on_event(StreamEvent(
                type="start",
                content=f"🎼 Orchestrator: {task[:80]}",
            ))
        return self.agent.run(task, stream=stream, on_event=self.on_event)

    def chat(self, message: str, stream: bool = True) -> str:
        """
        交互式对话 —— AgentPool 状态跨轮次保持。

        每一轮将当前子Agent状态注入上下文，
        让编排器能"看到"当前进展。
        """
        pool_state = self.pool.list_all()
        enriched = (
            f"{message}\n\n"
            f"[当前子Agent状态]\n{pool_state}"
        )
        return self.agent.run(enriched, stream=stream, on_event=self.on_event)

    # ── 快捷查询 ──────────────────────────────────
    def agent_status(self, name: str) -> str:
        return self.pool.status(name)

    def all_status(self) -> str:
        return self.pool.list_all()

    def wait(self, name: str, timeout: float | None = None) -> str:
        return self.pool.result(name, timeout=timeout)

    def wait_all(self, timeout: float | None = None) -> dict[str, str]:
        return self.pool.wait_all(timeout=timeout)
