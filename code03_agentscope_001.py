# intro agentscope react agent with kimi-k2.5 and a simple http tool

import asyncio
import os

from agentscope.agent import ReActAgent
from agentscope.mcp import HttpStatelessClient
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit
from pydantic import BaseModel, Field

from src.agent.agentscope_add import KimiChatFormatter
from src.config import LOCAL_IP


async def call_agent(prompt: str) -> dict:

    toolkit = Toolkit()

    simple_tool = HttpStatelessClient(
        # The name to identify the MCP
        name="simple_tool",
        transport="streamable_http",
        url=f"http://{LOCAL_IP}/mcpserver/simple_tool/mcp",
    )

    await toolkit.register_mcp_client(simple_tool)

    system_message = "你是良睦路程序员创建的一个机器人"
    agent = ReActAgent(
        name="小埋",
        sys_prompt=system_message,
        model=OpenAIChatModel(
            api_key=os.getenv("KIMI_API_KEY"),
            model_name="kimi-k2.5",
            stream=True,
            client_kwargs={"base_url": "https://api.moonshot.cn/v1"},
        ),
        formatter=KimiChatFormatter(),  # 使用自定义 formatter
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )

    msg = Msg("User", prompt, "user")
    result_msg = await agent(msg)
    return result_msg


def func1():

    res = asyncio.run(call_agent("现在几点了？"))
    print(res)


def func2():
    res = asyncio.run(call_agent("帮我查一下今天的天气"))
    print(res)


def func3():
    res = asyncio.run(call_agent("(1+3)*5等于多少"))
    print(res)



if __name__ == "__main__":
    func1()
    # func2()
    # func3()