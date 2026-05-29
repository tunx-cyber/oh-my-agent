"""
Todo 系统 —— Agent 的任务管理器。

设计理念（教育目的）:
  Agent 执行复杂任务时需要跟踪进度。这个 Todo 系统模拟了
  人类处理复杂任务的方式: 分解为子任务 → 逐个完成 → 标记状态。

特性:
  - 层级结构: 支持父子任务
  - 状态流转: pending → in_progress → completed (或 cancelled)
  - 自动序列化: 可注入 LLM 上下文作为进度提示
  - 幂等更新: 同一任务不会重复添加

用法:
    todos = TodoManager()
    todos.add("搜索最新论文", parent_id=None)
    todos.add("读取论文摘要", parent_id="search_papers")
    todos.start("search_papers")
    todos.complete("search_papers")
    print(todos.format_for_context())
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import uuid


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TodoItem:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    content: str = ""
    status: TodoStatus = TodoStatus.PENDING
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)  # child ids

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "children": self.children,
        }


class TodoManager:
    """
    层级任务管理器。

    关键设计: 同一时间只有一个任务处于 in_progress。
    这模拟了"专注一件事"的人类工作方式。
    """

    def __init__(self):
        self._items: dict[str, TodoItem] = {}

    # ── CRUD ──────────────────────────────────────
    def add(self, content: str, parent_id: str | None = None) -> str:
        """添加新任务，返回 task_id"""
        item = TodoItem(content=content, parent_id=parent_id)
        self._items[item.id] = item
        if parent_id and parent_id in self._items:
            self._items[parent_id].children.append(item.id)
        return item.id

    def update_status(self, task_id: str, status: TodoStatus):
        if task_id in self._items:
            self._items[task_id].status = status

    def start(self, task_id: str):
        """开始一个任务（同时只能有一个进行中）"""
        # 先完成其他进行中的任务
        for item in self._items.values():
            if item.status == TodoStatus.IN_PROGRESS:
                item.status = TodoStatus.PENDING
        self.update_status(task_id, TodoStatus.IN_PROGRESS)

    def complete(self, task_id: str):
        self.update_status(task_id, TodoStatus.COMPLETED)

    def cancel(self, task_id: str):
        self.update_status(task_id, TodoStatus.CANCELLED)

    def get(self, task_id: str) -> TodoItem | None:
        return self._items.get(task_id)

    # ── 查询 ──────────────────────────────────────
    def list_by_status(self, status: TodoStatus | None = None) -> list[TodoItem]:
        items = list(self._items.values())
        if status:
            items = [i for i in items if i.status == status]
        return sorted(items, key=lambda i: i.id)

    def current_task(self) -> TodoItem | None:
        for item in self._items.values():
            if item.status == TodoStatus.IN_PROGRESS:
                return item
        return None

    def progress(self) -> tuple[int, int]:
        """返回 (completed, total)"""
        total = len(self._items)
        completed = sum(1 for i in self._items.values() if i.status == TodoStatus.COMPLETED)
        return completed, total

    def is_all_done(self) -> bool:
        return all(
            i.status in (TodoStatus.COMPLETED, TodoStatus.CANCELLED)
            for i in self._items.values()
        )

    # ── 格式化（注入 LLM 上下文） ─────────────────
    def format_for_context(self) -> str:
        """
        将 todo 列表格式化为 LLM 可读的 markdown。
        这个输出会被注入到 system prompt 中，让 LLM 感知当前进度。
        """
        if not self._items:
            return ""

        lines = ["## 任务进度"]

        # 找根任务（没有 parent 的）
        roots = [i for i in self._items.values() if i.parent_id is None]
        for root in roots:
            lines.append(self._format_tree(root, indent=0))

        completed, total = self.progress()
        lines.append(f"\n*进度: {completed}/{total} 已完成*")
        return "\n".join(lines)

    def _format_tree(self, item: TodoItem, indent: int) -> str:
        prefix = "  " * indent
        status_icon = {
            TodoStatus.PENDING: "○",
            TodoStatus.IN_PROGRESS: "◉",
            TodoStatus.COMPLETED: "✓",
            TodoStatus.CANCELLED: "✗",
        }
        icon = status_icon.get(item.status, "?")
        line = f"{prefix}- {icon} [{item.id}] {item.content}"

        for child_id in item.children:
            if child_id in self._items:
                line += "\n" + self._format_tree(self._items[child_id], indent + 1)

        return line

    def to_list(self) -> list[dict]:
        return [item.to_dict() for item in self._items.values()]
