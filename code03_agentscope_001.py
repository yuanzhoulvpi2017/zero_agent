# intro agentscope react agent with kimi-k2.5 and a simple http tool

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.mcp import HttpStatelessClient
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit
from pydantic import BaseModel, Field
import os
import time
from src.config import LOCAL_IP
import asyncio


class KimiChatFormatter(OpenAIChatFormatter):
    """
    继承 OpenAIChatFormatter，修复 Kimi 开启 thinking 后
    assistant tool call 消息中缺少 reasoning_content 的问题。
    从原始 msgs 的 thinking block 中提取真实思考内容填充到格式化后的消息中。
    """

    async def _format(self, msgs):
        # 预先从原始 msgs 中提取每个含 tool_use 的 assistant 消息的 thinking 内容（按顺序）
        assistant_reasoning = []
        for msg in msgs:
            if msg.role == "assistant":
                thinking_text = ""
                has_tool_use = False
                for block in msg.get_content_blocks():
                    if block.get("type") == "thinking":
                        thinking_text = block.get("thinking", "")
                    elif block.get("type") == "tool_use":
                        has_tool_use = True
                if has_tool_use:
                    assistant_reasoning.append(thinking_text)

        messages = await super()._format(msgs)

        reasoning_iter = iter(assistant_reasoning)
        fixed_messages = []
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and msg.get("tool_calls")
                and "reasoning_content" not in msg
            ):
                msg = {**msg, "reasoning_content": next(reasoning_iter, "")}
            fixed_messages.append(msg)
        return fixed_messages


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
    agent.register_instance_hook(
        "pre_reply",
        hook_name="my_pre_reply_hook",
        hook=lambda self, kwargs: print(f"Pre-reply hook called with kwargs: {kwargs}"),
    )
    agent.register_instance_hook(
        "post_reply",
        hook_name="my_post_reply_hook",
        hook=lambda self, kwargs, output: print(f"Post-reply hook called with kwargs: {kwargs} and output: {output}"),
    )
    _reply_start_time = {}

    def pre_reply_timer(self, kwargs):
        _reply_start_time["t"] = time.perf_counter()

    def post_reply_timer(self, kwargs, output):
        elapsed = time.perf_counter() - _reply_start_time["t"]
        print(f"[计时] reply 耗时: {elapsed:.3f} 秒")

    agent.register_instance_hook("pre_reply", hook_name="timer_pre", hook=pre_reply_timer)
    agent.register_instance_hook("post_reply", hook_name="timer_post", hook=post_reply_timer)

    msg = Msg("User", prompt, "user")
    result_msg = await agent(msg)
    return result_msg


def func4():

    res = asyncio.run(call_agent_with_hook("现在几点了？"))
    print(res)


if __name__ == "__main__":
    func4()