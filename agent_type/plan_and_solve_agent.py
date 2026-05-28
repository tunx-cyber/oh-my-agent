# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划，一定要包含```python和```:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

from core.llm_mind import LLMMind
from typing import Optional
import ast
class Planner:
    def __init__(
        self,
        llm: LLMMind,
        prompt: Optional[str] = None
    ):
        self.llm = llm
        self.prompt = prompt
    
    def plan(self, question):
        prompt = self.prompt.format(question=question)
        messages = [{"role":"user","content":prompt}]
        response = self.llm.invoke(messages)
        print("plan为",response)

        try:
            plan = response.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan)
            return plan if isinstance(plan, list) else []
        except Exception as e:
            print("error in parsing plan")
            return []

class Executor:
    def __init__(
        self,
        llm: LLMMind,
        prompt: Optional[str] = None
    ):
        self.llm = llm
        self.prompt = prompt
    
    def execute(self, question, plan):
        history = ""
        final_ans = ""
        for i, step in enumerate(plan, 1):
            print(f"真正执行步骤{i}/{len(plan)}: ")
            prompt = self.prompt.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step = step
            )

            messages = [{"role":"user","content":prompt}]
            response = self.llm.invoke(messages) or ""

            response += f"步骤{i}\n结果: {response}\n"
            final_ans = response
            print(f"步骤{i} 已完成，结果为{response}")
        
        return final_ans
    
class MyPlanAndSolveAgent:
    def __init__(
        self,
        name:str,
        llm:LLMMind
    ):
        self.name = name
        self.llm = llm
        self.planner = Planner(self.llm,DEFAULT_PLANNER_PROMPT)
        self.executor = Executor(self.llm, DEFAULT_EXECUTOR_PROMPT)
    
    def run(self, input_text, **kwargs):
        plan = self.planner.plan(input_text)
        if not plan:
            print("nothing to plan")
        
        final_ans = self.executor.execute(input_text,plan, **kwargs)
        print("misson suceess")
        print(final_ans)
        return final_ans