"""
DeepAgents —— 教育性 Agent 框架演示

展示:
  1. DeepAgent:  规划 → 执行 → 自省 闭环
  2. Orchestrator: 多 Agent 编排（派发/查询/汇总）
  3. 流式输出:  实时看到 Agent 的思考和行动
"""

from core.llm_mind import LLMMind
from config.settings import get_settings
from agents.deep_agent import DeepAgent
from agents.orchestrator import OrchestratorAgent
from tools.builtin import read_file, write_file, list_directory, search_files, python_exec, web_search

settings = get_settings()
llm = LLMMind(
    model_name=settings.TEST_MODEL,
    api_key=settings.TEST_API_KEY,
    base_url=settings.TEST_API_URL,
    max_tokens=4096,
    provider="openai",
)

BASE_TOOLS = [read_file, write_file, list_directory, search_files, python_exec, web_search]


# ═══════════════════════════════════════════════════
# Demo 1: 单个 DeepAgent（基础用法）
# ═══════════════════════════════════════════════════
def demo_single_agent():
    """演示: 单个 DeepAgent 完成任务"""
    agent = DeepAgent(
        name="assistant",
        llm=llm,
        tools=BASE_TOOLS,
        max_retries=2,
    )
    return agent.run("查看当前目录有哪些 Python 文件，统计数量，保存到 py_count.txt")


# ═══════════════════════════════════════════════════
# Demo 2: 多 Agent 编排（一次性任务）
# ═══════════════════════════════════════════════════
def demo_orchestrator():
    """演示: Orchestrator 协调多个子 Agent"""
    orch = OrchestratorAgent(
        llm=llm,
        shared_tools=BASE_TOOLS,
    )
    return orch.run(
        "请完成以下任务:\n"
        "1. 派 searcher 子Agent 搜索 Python 3.12 的新特性\n"
        "2. 派 writer 子Agent 根据搜索结果写一份 markdown 报告\n"
        "3. 等两者都完成后，汇总结果"
    )


# ═══════════════════════════════════════════════════
# Demo 3: 交互式多 Agent（多轮对话）
# ═══════════════════════════════════════════════════
def demo_interactive():
    """演示: 交互式多 Agent —— 用户可随时询问子Agent进度"""
    orch = OrchestratorAgent(
        llm=llm,
        shared_tools=BASE_TOOLS,
    )

    print("╔══════════════════════════════════════════════╗")
    print("║  多 Agent 交互模式                          ║")
    print("║  输入 quit 退出                             ║")
    print("╚══════════════════════════════════════════════╝")

    while True:
        try:
            msg = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见")
            break

        if not msg or msg.lower() == "quit":
            print("👋 再见")
            break

        print()
        reply = orch.chat(msg)
        print(f"\n🤖 编排器: {reply}")


# ═══════════════════════════════════════════════════
# Demo 4: 并行子 Agent 对比
# ═══════════════════════════════════════════════════
def demo_parallel():
    """演示: 并行运行两个子Agent，对比结果"""
    orch = OrchestratorAgent(
        llm=llm,
        shared_tools=BASE_TOOLS,
    )

    # 第一轮: 派发两个子Agent
    orch.chat("派 coder1 用 python_exec 计算 1-100 质数之和")
    orch.chat("派 coder2 用 python_exec 计算 1-100 质数之和（用不同算法）")

    # 查询状态
    print("\n--- 查询状态 ---")
    print(orch.all_status())

    # 等待结果
    print("\n--- 等待结果 ---")
    r1 = orch.wait("coder1", timeout=60)
    r2 = orch.wait("coder2", timeout=60)
    print(f"coder1: {r1[:200]}")
    print(f"coder2: {r2[:200]}")

    # 汇总
    return orch.chat("两个coder的结果是否一致？请汇总")


if __name__ == "__main__":
    import sys

    demos = {
        "1": ("单个Agent", demo_single_agent),
        "2": ("多Agent编排", demo_orchestrator),
        "3": ("交互式多Agent", demo_interactive),
        "4": ("并行子Agent对比", demo_parallel),
    }

    print("选择演示:")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("> ").strip()

    if choice in demos:
        name, fn = demos[choice]
        print(f"\n{'='*60}")
        print(f"  运行: {name}")
        print(f"{'='*60}")
        result = fn()
        if result and choice != "3":
            print(f"\n{'='*60}")
            print(f"  最终结果:\n{result}")
            print(f"{'='*60}")
    else:
        print(f"未知: {choice}")
