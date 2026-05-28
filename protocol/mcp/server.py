"""
MCP Server 示例
提供三个工具：加法、获取当前时间、查询天气（模拟）
"""

import asyncio
import json
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# 创建 MCP Server 实例
app = Server("demo-server")


# ── 工具列表 ──────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """声明本 Server 提供哪些工具"""
    return [
        types.Tool(
            name="add",
            description="计算两个数字的和",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数"},
                    "b": {"type": "number", "description": "第二个数"},
                },
                "required": ["a", "b"],
            },
        ),
        types.Tool(
            name="get_time",
            description="获取当前服务器时间",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "时间格式，默认 '%Y-%m-%d %H:%M:%S'",
                    }
                },
            },
        ),
        types.Tool(
            name="get_weather",
            description="查询指定城市的天气（模拟数据）",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
        ),
    ]


# ── 工具实现 ──────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """处理工具调用请求"""

    if name == "add":
        a = arguments["a"]
        b = arguments["b"]
        result = a + b
        return [types.TextContent(type="text", text=f"{a} + {b} = {result}")]

    elif name == "get_time":
        fmt = arguments.get("format", "%Y-%m-%d %H:%M:%S")
        now = datetime.now().strftime(fmt)
        return [types.TextContent(type="text", text=f"当前时间：{now}")]

    elif name == "get_weather":
        city = arguments["city"]
        # 模拟天气数据
        mock_data = {
            "北京": {"temp": "15°C", "condition": "晴", "humidity": "30%"},
            "上海": {"temp": "22°C", "condition": "多云", "humidity": "65%"},
            "广州": {"temp": "28°C", "condition": "小雨", "humidity": "80%"},
            "新加坡": {"temp": "32°C", "condition": "雷阵雨", "humidity": "85%"},
        }
        weather = mock_data.get(city, {"temp": "未知", "condition": "无数据", "humidity": "未知"})
        text = (
            f"城市：{city}\n"
            f"温度：{weather['temp']}\n"
            f"天气：{weather['condition']}\n"
            f"湿度：{weather['humidity']}"
        )
        return [types.TextContent(type="text", text=text)]

    else:
        raise ValueError(f"未知工具：{name}")


# ── 资源列表（可选） ───────────────────────────────────────

@app.list_resources()
async def list_resources() -> list[types.Resource]:
    """声明本 Server 提供哪些资源"""
    return [
        types.Resource(
            uri="demo://readme",
            name="README",
            description="服务器说明文档",
            mimeType="text/plain",
        )
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "demo://readme":
        return "这是一个 MCP Demo Server，提供加法、时间查询和天气查询工具。"
    raise ValueError(f"未知资源：{uri}")


# ── 启动 ──────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())