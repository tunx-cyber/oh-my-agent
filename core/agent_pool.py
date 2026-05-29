"""
Agent Pool —— 子 Agent 的生命周期管理。

设计理念（教育目的）:
  多 Agent 协作的关键是"任务分发 + 状态同步"。
  主 Agent 将子任务派发给子 Agent，子 Agent 在后台线程中独立运行，
  主 Agent 通过 AgentPool 随时查询进度、获取结果。

架构:
  Orchestrator (主 Agent)
      │
      ├── spawn("researcher", task) → SubAgentHandle
      ├── spawn("coder", task)      → SubAgentHandle
      │
      ├── status("researcher")  → "running"
      ├── result("coder")       → (blocks until done)
      │
      └── list_all() → [researcher: running, coder: completed]

每个子 Agent 是一个 DeepAgent 实例，在独立线程中运行。
AgentPool 使用 threading.Lock 保证线程安全。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from threading import Thread, Lock
from typing import Callable
import time


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubAgentHandle:
    """指向一个正在/已经运行的子 Agent 的句柄"""
    name: str
    task: str = ""
    status: AgentStatus = AgentStatus.IDLE
    result: str = ""
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    _lock: Lock = field(default_factory=Lock)
    _thread: Thread | None = field(default=None, repr=False)

    def is_running(self) -> bool:
        return self.status == AgentStatus.RUNNING

    def is_done(self) -> bool:
        return self.status in (AgentStatus.COMPLETED, AgentStatus.FAILED)

    def elapsed(self) -> float:
        if self.started_at == 0:
            return 0
        end = self.finished_at if self.finished_at > 0 else time.time()
        return end - self.started_at

    def summary(self) -> str:
        """人类可读的状态摘要"""
        return (
            f"[{self.name}] {self.status.value} | "
            f"耗时: {self.elapsed():.1f}s | "
            f"任务: {self.task[:80]}"
            + (f" | 结果: {self.result[:100]}" if self.result else "")
        )


class AgentPool:
    """
    子 Agent 池 —— 管理多个并发运行的子 Agent。

    用法:
        pool = AgentPool()

        # 派发任务（子 Agent 在后台线程运行）
        pool.spawn(
            name="researcher",
            agent_factory=lambda: DeepAgent(name="researcher", llm=llm, tools=[search]),
            task="搜索 2024 年 AI 论文",
        )

        # 随时查询状态
        print(pool.status("researcher"))

        # 等待结果
        result = pool.result("researcher", timeout=60)

        # 列出所有子 Agent
        print(pool.list_all())
    """

    def __init__(self):
        self._agents: dict[str, SubAgentHandle] = {}
        self._lock = Lock()

    # ── 派发 ──────────────────────────────────────
    def spawn(
        self,
        name: str,
        agent_factory: Callable[[], object],  # () -> DeepAgent
        task: str,
        on_event: Callable | None = None,
    ) -> SubAgentHandle:
        """
        派发任务给子 Agent，后台运行。

        agent_factory: 返回 DeepAgent 实例的工厂函数（延迟构造）
        task: 子 Agent 要执行的任务描述
        """
        with self._lock:
            if name in self._agents and self._agents[name].is_running():
                raise RuntimeError(f"Agent '{name}' 已经在运行中")

            handle = SubAgentHandle(name=name, task=task, status=AgentStatus.RUNNING)
            self._agents[name] = handle

        def _run():
            handle.started_at = time.time()
            try:
                agent = agent_factory()
                result = agent.run(task, stream=False)
                with handle._lock:
                    handle.result = result or ""
                    handle.status = AgentStatus.COMPLETED
            except Exception as e:
                with handle._lock:
                    handle.error = str(e)
                    handle.status = AgentStatus.FAILED
            finally:
                handle.finished_at = time.time()

        thread = Thread(target=_run, name=f"subagent-{name}", daemon=True)
        handle._thread = thread
        thread.start()

        return handle

    # ── 查询 ──────────────────────────────────────
    def status(self, name: str) -> str:
        """获取子 Agent 状态摘要"""
        handle = self._agents.get(name)
        if handle is None:
            return f"[{name}] 不存在"
        return handle.summary()

    def result(self, name: str, timeout: float | None = None) -> str:
        """
        获取子 Agent 结果（阻塞直到完成或超时）。

        返回: 子 Agent 的 final_answer，或超时时的当前状态
        """
        handle = self._agents.get(name)
        if handle is None:
            return f"[错误] Agent '{name}' 不存在"

        if handle.is_done():
            return handle.result if handle.status == AgentStatus.COMPLETED else f"[失败] {handle.error}"

        if handle._thread:
            handle._thread.join(timeout=timeout)

        if handle.is_done():
            return handle.result if handle.status == AgentStatus.COMPLETED else f"[失败] {handle.error}"
        else:
            return f"[运行中] {handle.summary()}"

    def get_handle(self, name: str) -> SubAgentHandle | None:
        return self._agents.get(name)

    # ── 列表 ──────────────────────────────────────
    def list_all(self) -> str:
        """列出所有子 Agent 及其状态"""
        if not self._agents:
            return "(无子 Agent)"

        lines = [f"📊 子 Agent 池 ({len(self._agents)} 个):"]
        for name, handle in self._agents.items():
            icon = {
                AgentStatus.IDLE: "○",
                AgentStatus.RUNNING: "◉",
                AgentStatus.COMPLETED: "✓",
                AgentStatus.FAILED: "✗",
            }.get(handle.status, "?")
            lines.append(f"  {icon} {handle.summary()}")
        return "\n".join(lines)

    def all_done(self) -> bool:
        """所有子 Agent 是否都已完成"""
        return all(h.is_done() for h in self._agents.values())

    def wait_all(self, timeout: float | None = None) -> dict[str, str]:
        """等待所有子 Agent 完成，返回 {name: result}"""
        results = {}
        for name in list(self._agents.keys()):
            results[name] = self.result(name, timeout=timeout)
        return results
