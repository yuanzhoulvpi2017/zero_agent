from src.agent.simple_agent import typewriter_noprint
from qwen_agent.agents import Assistant
import os
from src.config import LOCAL_IP

llm_cfg = {
    "model": "kimi-k2.5",
    "api_key": os.getenv("KIMI_API_KEY"),
    "model_server": "https://api.moonshot.cn/v1",
}

tools = [
    {
        "mcpServers": {
            "simple_tool": {
                "url": f"http://{LOCAL_IP}/mcpserver/simple_tool/mcp",
                "type": "streamable-http",
            },
        }
    }
]
SYS_MESSAGE = """你叫小埋，由B站UP主(良睦路程序员),创建。是一个聊天机器人，你的任务是和用户进行自然、有趣、友好的聊天互动。"""


def base_agent_(llm_cfg, messages):

    agent = Assistant(llm=llm_cfg, function_list=tools, system_message=SYS_MESSAGE)

    response_plain_text = ""

    for ret_messages in agent.run(messages):
        response_plain_text = typewriter_noprint(ret_messages, response_plain_text)

    return response_plain_text


def test001():

    test_message = [{"role": "user", "content": "你好！你是谁"}]

    result = base_agent_(llm_cfg, messages=test_message)
    print(result)


def test002():
    test_message = [{"role": "user", "content": "现在是几点"}]

    result = base_agent_(llm_cfg, messages=test_message)
    print(result)


def test003():
    test_message = [{"role": "user", "content": "1+ 3等于多少"}]

    result = base_agent_(llm_cfg, messages=test_message)
    print(result)


test003()
