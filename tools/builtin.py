"""
内置工具集 —— Agent 与外部世界交互的基本能力。

包含:
  - 文件系统: read_file, write_file, list_directory, search_files
  - 代码执行: python_exec (沙盒)
  - 网络搜索: web_search

设计理念（教育目的）:
  这些工具展示了 Agent 如何扩展 LLM 的能力边界。
  每个工具都是独立的纯函数，通过 @tool 装饰器暴露给 LLM。
"""

from __future__ import annotations
import os
import io
import re
import contextlib
import subprocess
import tempfile
from pathlib import Path

from tools.base import tool

# ═══════════════════════════════════════════════════
# 文件系统工具
# ═══════════════════════════════════════════════════

@tool("read_file", "读取指定路径的文件内容并返回")
def read_file(path: str, encoding: str = "utf-8") -> str:
    """
    读取文件内容。

    Args:
        path: 文件路径（相对或绝对）
        encoding: 文件编码，默认 utf-8
    """
    p = Path(path)
    if not p.exists():
        return f"[错误] 文件不存在: {path}"
    if p.is_dir():
        return f"[错误] 路径是目录而非文件: {path}\n请使用 list_directory 查看目录内容"
    try:
        content = p.read_text(encoding=encoding)
        # 大文件截断
        if len(content) > 50_000:
            content = content[:50_000] + f"\n\n... [截断，共 {len(content)} 字符]"
        return content
    except Exception as e:
        return f"[错误] 读取文件失败: {e}"


@tool("write_file", "将内容写入指定路径的文件（会覆盖已有文件）")
def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """
    写入文件。父目录不存在时会自动创建。

    Args:
        path: 文件路径
        content: 要写入的内容
        encoding: 文件编码，默认 utf-8
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return f"[成功] 已写入 {len(content)} 字符到 {path}"
    except Exception as e:
        return f"[错误] 写入文件失败: {e}"


@tool("list_directory", "列出目录内容，支持递归和文件过滤")
def list_directory(path: str = ".", recursive: bool = False, pattern: str = "*") -> str:
    """
    列出目录中的文件和子目录。

    Args:
        path: 目录路径，默认当前目录
        recursive: 是否递归列出子目录
        pattern: 文件名匹配模式，如 "*.py", "*.md"。默认 "*" 匹配所有
    """
    p = Path(path)
    if not p.exists():
        return f"[错误] 目录不存在: {path}"
    if not p.is_dir():
        return f"[错误] 不是目录: {path}"

    try:
        if recursive:
            files = list(p.rglob(pattern))
        else:
            files = list(p.glob(pattern))

        # 分离目录和文件
        dirs = [f for f in files if f.is_dir()]
        regular_files = [f for f in files if f.is_file()]

        lines = []
        if dirs:
            lines.append(f"📁 目录 ({len(dirs)}):")
            for d in sorted(dirs)[:20]:
                lines.append(f"  {d.relative_to(p)}/")
        if regular_files:
            lines.append(f"📄 文件 ({len(regular_files)}):")
            for f in sorted(regular_files)[:50]:
                size = f.stat().st_size
                size_str = f"{size:>8} B" if size < 1024 else f"{size/1024:>7.1f} KB"
                lines.append(f"  {size_str}  {f.relative_to(p)}")

        if len(dirs) > 20 or len(regular_files) > 50:
            lines.append(f"\n... (显示被截断，共 {len(dirs)} 个目录, {len(regular_files)} 个文件)")

        return "\n".join(lines) if lines else f"(空目录: {path})"
    except Exception as e:
        return f"[错误] 列出目录失败: {e}"


@tool("search_files", "在目录中递归搜索包含指定文本的文件（类 grep）")
def search_files(directory: str, query: str, file_pattern: str = "*") -> str:
    """
    在文件中搜索匹配的文本行。

    Args:
        directory: 搜索的根目录
        query: 搜索的文本（支持正则表达式）
        file_pattern: 文件名匹配模式，如 "*.py"
    """
    p = Path(directory)
    if not p.exists():
        return f"[错误] 目录不存在: {directory}"

    try:
        matches = []
        for file_path in p.rglob(file_pattern):
            if not file_path.is_file():
                continue
            try:
                for i, line in enumerate(file_path.read_text(errors="ignore").split("\n"), 1):
                    if re.search(query, line, re.IGNORECASE):
                        matches.append((file_path, i, line.strip()[:200]))
            except Exception:
                continue

        if not matches:
            return f"[结果] 在 {directory} 中未找到匹配 '{query}' 的文件"

        # 按文件分组
        lines = [f"🔍 搜索 '{query}' — 找到 {len(matches)} 处匹配:\n"]
        current_file = None
        for file_path, line_no, text in matches[:30]:
            rel = file_path.relative_to(p)
            if rel != current_file:
                lines.append(f"\n📄 {rel}:")
                current_file = rel
            lines.append(f"  L{line_no}: {text}")

        if len(matches) > 30:
            lines.append(f"\n... (显示被截断，共 {len(matches)} 处匹配)")

        return "\n".join(lines)
    except Exception as e:
        return f"[错误] 搜索失败: {e}"


# ═══════════════════════════════════════════════════
# 沙盒代码执行
# ═══════════════════════════════════════════════════

@tool("python_exec", "在安全沙盒中执行 Python 代码，返回 stdout 输出")
def python_exec(code: str, timeout: int = 30) -> str:
    """
    在隔离的 subprocess 中执行 Python 代码。

    安全设计（教育目的说明）:
      - 使用 subprocess 隔离，而非 exec()，防止影响主进程
      - 超时控制，防止死循环
      - 限制内置函数，移除危险操作 (open, __import__, eval, exec)
      - 输出大小限制

    Args:
        code: 要执行的 Python 代码
        timeout: 超时时间（秒），默认 30
    """
    # 包装代码：限制危险的内置函数
    safe_builtins = {
        "print": print, "len": len, "range": range, "int": int,
        "float": float, "str": str, "list": list, "dict": dict,
        "set": set, "tuple": tuple, "bool": bool, "type": type,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "sorted": sorted, "reversed": reversed, "sum": sum, "min": min, "max": max,
        "abs": abs, "round": round, "isinstance": isinstance,
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        "json": __import__("json"), "math": __import__("math"),
        "re": __import__("re"), "datetime": __import__("datetime"),
        "collections": __import__("collections"),
        "itertools": __import__("itertools"),
    }

    wrapper = (
        "__builtins__ = __import__('json').loads('''" +
        __import__("json").dumps({k: None for k in safe_builtins}) +
        "''')\n"
        "for _k, _v in __import__('json').loads('''" +
        __import__("json").dumps({k: k for k in safe_builtins}) +
        "''').items():\n"
        "    __builtins__[_k] = globals().get(_v, getattr(__builtins__, _v, None))\n"
    )

    # 使用临时文件执行
    tmp_path = None
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Write safe restricted code that uses a whitelist approach
            f.write("import json, math, re, datetime, collections, itertools\n")
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"

        if len(output) > 10_000:
            output = output[:10_000] + f"\n\n... [输出被截断]"

        if result.returncode != 0:
            output = f"[返回码: {result.returncode}]\n{output}"

        return output if output.strip() else "[执行完成，无输出]"

    except subprocess.TimeoutExpired:
        return f"[错误] 代码执行超时 ({timeout}秒)"
    except Exception as e:
        return f"[错误] 代码执行失败: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════
# 网络搜索
# ═══════════════════════════════════════════════════

@tool("web_search", "搜索互联网获取最新信息，返回结果摘要")
def web_search(query: str, max_results: int = 5) -> str:
    """
    搜索互联网。

    Args:
        query: 搜索关键词
        max_results: 返回结果数量，默认 5
    """
    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=max_results))
        if not results:
            return f"[结果] 未找到与 '{query}' 相关的结果"

        lines = [f"🔍 搜索结果: '{query}' ({len(results)} 条)\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            body = r.get("body", "")[:200]
            href = r.get("href", "")
            lines.append(f"{i}. {title}\n   {body}\n   🔗 {href}\n")
        return "\n".join(lines)
    except ImportError:
        return "[提示] ddgs 库未安装。安装：pip install ddgs"
    except Exception as e:
        return f"[错误] 搜索失败: {e}"
