from typing import List

from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import ASSISTANT, FUNCTION
from qwen_agent.utils.output_beautify import (
    ANSWER_S,
    THOUGHT_S,
    TOOL_CALL_S,
    TOOL_RESULT_S,
)

from ..config import LOCAL_IP


# copy code from qwen_agent.utils.output_beautify import typewriter_print
def typewriter_noprint(messages: List[dict], text: str) -> str:
    full_text = ""
    content = []
    for msg in messages:
        if msg["role"] == ASSISTANT:
            if msg.get("reasoning_content"):
                assert isinstance(msg["reasoning_content"], str), (
                    "Now only supports text messages"
                )
                content.append(f"{THOUGHT_S}\n{msg['reasoning_content']}")
            if msg.get("content"):
                assert isinstance(msg["content"], str), (
                    "Now only supports text messages"
                )
                content.append(f"{ANSWER_S}\n{msg['content']}")
            if msg.get("function_call"):
                content.append(
                    f"{TOOL_CALL_S} {msg['function_call']['name']}\n{msg['function_call']['arguments']}"
                )
        elif msg["role"] == FUNCTION:
            content.append(f"{TOOL_RESULT_S} {msg['name']}\n{msg['content']}")
        else:
            raise TypeError
    if content:
        full_text = "\n".join(content)

    return full_text


tools = [
    {
        "simple_tool": {
            "url": f"http://{LOCAL_IP}/mcpserver/simple_tool/mcp",
            "type": "streamable-http",
        },
    }
]
SYS_MESSAGE = """你叫小埋，由B站UP主(良睦路程序员),创建。是一个聊天机器人，你的任务是和用户进行自然、有趣、友好的聊天互动。"""


def base_agent_(llm_cfg, messages):
    agent = Assistant(llm=llm_cfg, function_list=tools, system_message=SYS_MESSAGE)

    response_plain_text = ""

    for ret_messages in agent.run(messages):
        response_plain_text = typewriter_noprint(ret_messages, response_plain_text)

    return response_plain_text
