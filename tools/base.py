'''
tool装饰器，注册一个函数为一个tool
'''
from typing import Dict, Callable, Optional, Any, get_type_hints,overload
import inspect
import re
from pydantic import BaseModel

class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

class Tool:
    def __init__(self,func: Callable, name: str = None, description: str = None,):
        self.func = func
        if name == None:
            self.name = func.__name__
        else:
            self.name = name
        if description == None:
            self.description = self._extract_description_from_docstring()
        else:
            self.description = description

        self.parameters = self._parse_parameters()

    def __call__(self, *args, **kwargs) -> Any:
        """让Tool实例可以像函数一样被调用"""
        return self.func(*args, **kwargs)
    
    def _parse_parameters(self):
        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)
        docstring = inspect.getdoc(self.func) or ""
        # 从 docstring 解析参数描述
        param_descriptions = self._parse_param_descriptions(docstring)

        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            # 获取类型
            param_type_hint = type_hints.get(param_name, str)
            param_type = self._python_type_to_tool_type(param_type_hint)

            # 判断是否必需
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default

            # 获取描述
            description = param_descriptions.get(param_name, f"参数 {param_name}")

            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=description,
                required=required,
                default=default
            ))

        return parameters

    def _python_type_to_tool_type(self, py_type) -> str:
        """将 Python 类型转换为工具类型字符串"""
        # 处理泛型类型
        origin = getattr(py_type, '__origin__', None)
        if origin is not None:
            if origin is list:
                return "array"
            elif origin is dict:
                return "object"

        # 处理基本类型
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        return type_map.get(py_type, "string")
    
    def _extract_description_from_docstring(self) -> str:
        """从 docstring 提取描述"""
        doc = inspect.getdoc(self.func)
        if not doc:
            return f"{self.func.__name__}"

        # 提取第一行作为描述
        lines = doc.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Args:') and not line.startswith('Returns:'):
                return line

        return f"{self.func.__name__}"
    
    def _parse_param_descriptions(self, docstring: str) -> Dict[str, str]:
        """
        从 docstring 解析参数描述

        Args:
            param_name: 参数描述
            another_param: 另一个参数描述
        """
        descriptions = {}

        # 查找 Args: 部分
        args_match = re.search(r'Args:\s*\n(.*?)(?:\n\s*\n|Returns:|$)', docstring, re.DOTALL)
        if not args_match:
            return descriptions

        args_section = args_match.group(1)

        # 解析每个参数
        # 匹配格式: param_name: 描述 或 param_name (type): 描述
        param_pattern = r'^\s*(\w+)(?:\s*\([^)]+\))?\s*:\s*(.+?)(?=^\s*\w+\s*(?:\([^)]+\))?\s*:|$)'
        matches = re.finditer(param_pattern, args_section, re.MULTILINE | re.DOTALL)

        for match in matches:
            param_name = match.group(1).strip()
            param_desc = match.group(2).strip()
            # 清理描述中的多余空白
            param_desc = re.sub(r'\s+', ' ', param_desc)
            descriptions[param_name] = param_desc

        return descriptions
    def get_parameters(self):
        return self.parameters
    
    def __repr__(self) -> str:
        return f"Tool(name={self.name!r}, description={self.description!r})"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [param.dict() for param in self.get_parameters()]
        }

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling schema 格式

        用于 FunctionCallAgent，使工具能够被 OpenAI 原生 function calling 使用

        Returns:
            符合 OpenAI function calling 标准的 schema
        """
        parameters = self.get_parameters()

        # 构建 properties
        properties = {}
        required = []

        for param in parameters:
            # 基础属性定义
            prop = {
                "type": param.type,
                "description": param.description
            }

            # 如果有默认值，添加到描述中（OpenAI schema 不支持 default 字段）
            if param.default is not None:
                prop["description"] = f"{param.description} (默认: {param.default})"

            # 如果是数组类型，添加 items 定义
            if param.type == "array":
                prop["items"] = {"type": "string"}  # 默认字符串数组

            properties[param.name] = prop

            # 收集必需参数
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def __str__(self) -> str:
        return f"Tool(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()

def tool(
        name: str | None = None,
        description: Optional[str] = None
    )->Tool:
    def _create_tool_factory(
        name: str,
    ):
        def _tool_factory(dec_func: Callable) ->Tool:
            return Tool(
                dec_func,
                name,
                description
            )
        return _tool_factory
    def _partial(func: Callable) -> Tool:
        """Partial function that takes a callable and returns a tool."""
        name_ = name if name else func.__name__
        tool_factory = _create_tool_factory(name_)
        return tool_factory(func)

    return _partial
        
class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool | Callable):
        if isinstance(tool, Tool):
            tool_name = tool.name
            self.tools[tool_name] = tool
        elif isinstance(tool, Callable):
            tool_obj = Tool(func=tool)
            tool_name = tool_obj.name
            self.tools[tool_name] = tool
        else:
            raise Exception("tool is not Callable or Tool")
        
    def get_tool(self, name: str):
        if name in self.tools.keys():
            return self.tools[name]
        else:
            raise Exception("No tool found")
        
    def execute_tool(self, name, **params):
        if name in self.tools.keys():
            return self.tools[name](**params)
        else:
            raise Exception("No tool found")
        
    def get_tools_description(self) -> str:
        descriptions = []
        for name,tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions) if descriptions else "暂无可用工具"
    
    def list_tools(self) -> list[str]:
        return list(self.tools.keys())
    
    def clear(self):
        self.tools.clear()