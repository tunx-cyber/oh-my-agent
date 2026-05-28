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