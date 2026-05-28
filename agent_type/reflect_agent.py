DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}
from core.llm_mind import LLMMind
from typing import Optional
class Memory:
    def __init__(self):
        self.records = []
    
    def add_record(self, type, content):
        self.records.append(
            {
                "type": type,
                "content":content
            }
        )
    def get_last_execution(self):
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
            
        return ""
    
    def get_trajectory(self):
        trajectory = ""
        for record in self.records:
            if record["type"] == "execution":
                trajectory += f"---上一轮尝试---\n{record["content"]}\n\n"
            elif record["type"] == "reflection":
                trajectory += f"---评审员反馈---\n{record["content"]}\n\n"
        return trajectory

class MyReflectionAgent:
    def __init__(
        self,
        name: str,
        llm: LLMMind,
        system_prompt: Optional[str] = None,
        max_iter: int  = 3,
        prompts: dict = DEFAULT_PROMPTS
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_iter = max_iter
        self.prompts = prompts
    
    def run(self, input_text, **kwargs):
        self.memory = Memory()
        print("initial attemp")
        initial_attemp = self.prompts["initial"].format(task = input_text)
        initial_result = self._get_llm_response(initial_attemp)
        self.memory.add_record("execution",initial_result)
        print("初始结果",initial_result)
        for i in range(self.max_iter):
            print(f"第{i+1}/{self.max_iter}轮迭代")
            print("真正反思...")
            last_result = self.memory.get_last_execution()
            reflection_prompt = self.prompts["reflect"].format(
                task = input_text,
                content=last_result
            )
            feedback = self._get_llm_response(reflection_prompt,**kwargs)
            self.memory.add_record("reflection",feedback)
            print(feedback)
            if "无需改进" in feedback in feedback.lower():
                print("反思结果完成，任务结束")
                break

            print("正在优化")
            refine_prompt = self.prompts["refine"].format(
                task = input_text,
                last_attempt = last_result,
                feedback = feedback
            )
            refine_result = self._get_llm_response(refine_prompt,**kwargs)
            print(refine_result)
            self.memory.add_record("execution",refine_result)
        
        final_result = self.memory.get_last_execution()
        print("最终结果为",final_result)
        return final_result

    def _get_llm_response(self, prompt, **kwargs):
        messages = [{"role":"user","content":prompt}]
        return self.llm.invoke(messages, **kwargs)
    
