import asyncio
import os
import time

from agentscope.agent import ReActAgent
from agentscope.mcp import HttpStatelessClient
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

from src.agent.agentscope_add import KimiChatFormatter
from src.config import LOCAL_IP


async def call_agent_with_hook(prompt: str) -> dict:

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
    # agent.register_instance_hook(
    #     "pre_reply",
    #     hook_name="my_pre_reply_hook",
    #     hook=lambda self, kwargs: print(f"Pre-reply hook called with kwargs: {kwargs}"),
    # )
    # agent.register_instance_hook(
    #     "post_reply",
    #     hook_name="my_post_reply_hook",
    #     hook=lambda self, kwargs, output: print(
    #         f"Post-reply hook called with kwargs: {kwargs} and output: {output}"
    #     ),
    # )
    _reply_start_time = {}

    def pre_reply_timer(self, kwargs):
        _reply_start_time["t"] = time.perf_counter()

    def post_reply_timer(self, kwargs, output):
        elapsed = time.perf_counter() - _reply_start_time["t"]
        print(f"[计时] reply 耗时: {elapsed:.3f} 秒")

    agent.register_instance_hook(
        "pre_reply", hook_name="timer_pre", hook=pre_reply_timer
    )
    agent.register_instance_hook(
        "post_reply", hook_name="timer_post", hook=post_reply_timer
    )

    msg = Msg("User", prompt, "user")
    result_msg = await agent(msg)
    return result_msg


def func4():

    res = asyncio.run(call_agent_with_hook("现在几点了？"))
    print(res)


if __name__ == "__main__":
    func4()
