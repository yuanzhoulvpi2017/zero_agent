from qwen_agent.llm.schema import (
    CONTENT,
    DEFAULT_SYSTEM_MESSAGE,
    ROLE,
    SYSTEM,
    USER,
    ContentItem,
    Message,
)
import json
import os
from qwen_agent.llm import get_chat_model

llm_cfg = {
    "model": "kimi-k2.5",
    "api_key": os.getenv("KIMI_API_KEY"),
    "model_server": "https://api.moonshot.cn/v1",
}


llm = get_chat_model(cfg=llm_cfg)


# def test001():

#     prompt = "你是谁"
#     *_, responses = llm.chat(messages=[Message(role=USER, content=prompt)])
#     print(responses)


def test002():
    messages = [
        {
            "role": "system",
            "content": "你叫小埋，由B站UP主(良睦路程序员),创建。是一个聊天机器人，你的任务是和用户进行自然、有趣、友好的聊天互动。",
        },
        {"role": "user", "content": "1+ 3等于多少"},
    ]

    with open("raw_data/function.json", "r") as f:
        functions = json.load(f)

    res = []
    for output in llm.chat(
        messages=messages,
        functions=functions,
        stream=False,
        extra_generate_cfg={"lang": "zh"},
    ):
        print(output)
        res.append(output)

    return res


test002()
