# Lifetime of an Agent(从0开始的agent)

## Motivation
LLMs are great! Agents are fantastic! But we don’t know what exactly happens inside such a black box. This project aims to reveal that black box.

大语言模型很牛！智能体更牛逼！但我们不知道这个黑盒内部到底发生了什么。这个项目旨在打开这个黑盒。

## Project Framework
- 从0构造LLM，并且可以训练
- 从0构造最小agent，并且可以使用
```text
myagent/
├── core/                        # 核心模块（Agent 的"大脑"）
│   ├── llm_mind.py             # LLM 客户端（直接 openai.OpenAI）
│   ├── embedding.py            # Embedding 客户端
│   ├── messages.py             # 消息类型
│   ├── exceptions.py           # 自定义异常
│   ├── state.py                # Agent 状态（PlanStep, AgentState...）
│   ├── engine.py               # 编排引擎（显式状态机，替代 LangGraph）
│   ├── context.py              # 上下文管理器（GSSC: 收集→筛选→组织→压缩）
│   ├── memory.py               # 三层记忆（工作/情节/语义）
│   └── todo.py                 # 层级任务跟踪
│
├── agents/                      # Agent 节点（规划→执行→自省的组件）
│   ├── planner.py              # 将任务分解为原子步骤
│   ├── executor.py             # 执行步骤（工具调用/推理/综合）
│   ├── reflector.py            # 质量门控（置信度门控 + 自动升级）
│   ├── router.py               # 条件路由决策
│   └── deep_agent.py           # 完整 Agent：集成所有模块
│
├── tools/                       # 工具系统（Agent 的"双手"）
│   ├── base.py                 # Tool / ToolRegistry（LangChain 风格）
│   └── builtin.py              # 内置工具（文件系统、代码沙盒、搜索）
│
├── agent_type/                  # 保留：经典 agent 模式示例
│   ├── react_agent.py          # ReAct (Reasoning + Acting)
│   ├── plan_and_solve_agent.py # Plan & Solve
│   ├── reflect_agent.py        # Reflection
│   └── advance_agent.py        # 旧版 deep agent（保留作参考）
│                   
├── memory/                     # 向量存储（faiss, milvus...）
├── config/                     # 配置
├── tests/                      # 测试
└── main.py                     # 入口 & 演示
```
- main.py 是一个例子
## Statement
- 这个项目是比较学术化的，旨在帮助理解运作原理。
- 如果你喜欢，请给个star吧！

## TODO

1. search 更加完整，可以访问里面的网页
2. 编写最新架构的小llm
3. 上下文管理，记忆管理
4. rag
5. 安全代码执行沙盒
6. skill

