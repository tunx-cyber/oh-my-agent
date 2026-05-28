MY_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:
- `ToolCall[{{"tool_name":你要调用的工具名,"tool_params":你要调用的工具的参数字典}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:上面描述的json格式
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""

# my_react_agent.py

import re
from typing import Optional, List, Tuple
from core.llm_mind import LLMMind
from tools.base import Tool, ToolRegistry
from core.messages import Message
class MyReActAgent:
    """
    重写的ReAct Agent - 推理与行动结合的智能体
    """

    def __init__(
        self,
        name: str,
        llm: LLMMind,
        tools: Optional[Tool] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        self.name = name
        self.llm = llm
        # self.tools = tools
        self.system_prompt = system_prompt
        self.tool_registry = ToolRegistry()
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else MY_REACT_PROMPT
        if tools:
            for tool in tools:
                print(f"注册工具[{tool.name}]")
                self.tool_registry.register_tool(tool)
        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")
    
    def add_message(self, message):
        self.current_history.append(message)

    def _parse_output(self, text):
        import re
        # 使用正则表达式更精确地匹配Thought和Action
        thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', text, re.DOTALL)
        action_match = re.search(r'Action:\s*(.*)', text, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""
        
        return thought, action
    
    def _parse_action(self, text):
        import json
        import re
        # 使用正则表达式提取ToolCall中的JSON
        match = re.search(r'ToolCall\[(\{.*?\})\]', text, re.DOTALL)
        if not match:
            raise ValueError(f"无法解析工具调用格式: {text}")
        try:
            obj = json.loads(match.group(1))
            tool_name = obj.get("tool_name")
            tool_params = obj.get("tool_params", {})
            if not tool_name:
                raise ValueError("缺少tool_name")
            return tool_name, tool_params
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析错误: {e}")
        
    def _parse_action_input(self, text):
        import re
        # 使用正则表达式提取Finish中的答案
        match = re.search(r'Finish\[(.+)\]', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            # 如果格式不匹配，返回整个text作为答案
            return text.strip()
    def run(self, input_text: str, **kwargs) -> str:
        """运行ReAct Agent"""
        self.current_history = []
        current_step = 0

        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 1. 构建提示词
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )

            # 2. 调用LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages, **kwargs)
            # print(response_text)
            # 3. 解析输出
            try:
                thought, action = self._parse_output(response_text)
                print(f"Thought: {thought}")
                print(f"Action: {action}")
            except Exception as e:
                print(f"解析输出失败: {e}")
                # 如果解析失败，尝试继续或返回错误
                final_answer = "解析响应失败，请检查LLM输出格式。"
                return final_answer
            
            # 4. 检查完成条件
            if action and action.startswith("Finish"):
                try:
                    final_answer = self._parse_action_input(action)
                    self.add_message(Message(input_text, "user"))
                    self.add_message(Message(final_answer, "assistant"))
                    return final_answer
                except Exception as e:
                    print(f"解析最终答案失败: {e}")
                    final_answer = "无法解析最终答案。"
                    return final_answer

            # 5. 执行工具调用
            if action:
                try:
                    tool_name, tool_input = self._parse_action(action)
                    print(f"调用工具: {tool_name} 参数: {tool_input}")
                    observation = self.tool_registry.execute_tool(tool_name, **tool_input)
                    self.current_history.append(f"Action: {action}")
                    self.current_history.append(f"Observation: {observation}")
                except Exception as e:
                    print(f"工具调用失败: {e}")
                    observation = f"工具调用错误: {str(e)}"
                    self.current_history.append(f"Action: {action}")
                    self.current_history.append(f"Observation: {observation}")

        # 达到最大步数
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer
