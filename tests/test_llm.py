from core.llm_mind import LLMMind
from config.settings import get_settings

s = get_settings()
client = LLMMind(
    model_name=s.TEST_MODEL,
    api_key=s.TEST_API_KEY,
    base_url=s.TEST_API_URL,
    max_tokens=3200,
    provider="openai"
)

from agent_type.react_agent import MyReActAgent
from tools.build_in.terminal import TerminalTool
from tools.base import tool

@tool("terminal")
def terminal(cmd):
    f'''
    {TerminalTool().get_parameters()}
    '''
    tm = TerminalTool()
    return tm.run({"command":cmd})

agent = MyReActAgent("search",client,[terminal])
agent.run("当前目录有哪些文件")


# from agents.plan_and_solve_agent import MyPlanAndSolveAgent
# agent = MyPlanAndSolveAgent("test",client)
# # agent.run("一个圆形花园的半径是8米，如果要在花园周围建一条宽2m的小径，小径的面积是多少")
# from agents.reflect_agent import MyReflectionAgent
# agent = MyReflectionAgent("反思助手",client)
# result = agent.run("写一篇关于人工智能发展历程的简短文章")
# print(f"最终结果: {result}")