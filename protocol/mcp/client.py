"""
MCP Client 示例
以子进程方式启动 mcp_server.py，然后调用其工具
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # ── 连接到 Server ──────────────────────────────────────
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],   # Server 脚本路径
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # 初始化握手
            await session.initialize()
            print("✅ 已连接到 MCP Server\n")

            # ── 列出可用工具 ────────────────────────────────
            tools_response = await session.list_tools()
            print("📦 可用工具：")
            for tool in tools_response.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()

            # ── 列出可用资源 ────────────────────────────────
            resources_response = await session.list_resources()
            print("📄 可用资源：")
            for res in resources_response.resources:
                print(f"  - {res.uri}: {res.name}")
            print()

            # ── 读取资源 ────────────────────────────────────
            readme = await session.read_resource("demo://readme")
            print(f"📖 README 内容：{readme.contents[0].text}\n")

            # ── 调用工具：加法 ──────────────────────────────
            print("🔧 调用工具 [add] ...")
            result = await session.call_tool("add", {"a": 42, "b": 58})
            print(f"   结果：{result.content[0].text}\n")

            # ── 调用工具：获取时间 ───────────────────────────
            print("🔧 调用工具 [get_time] ...")
            result = await session.call_tool("get_time", {})
            print(f"   结果：{result.content[0].text}\n")

            # ── 调用工具：查询天气 ───────────────────────────
            for city in ["北京", "新加坡", "纽约"]:
                print(f"🔧 调用工具 [get_weather] 城市={city} ...")
                result = await session.call_tool("get_weather", {"city": city})
                print(f"   结果：\n{result.content[0].text}\n")


if __name__ == "__main__":
    asyncio.run(main())