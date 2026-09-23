"""使用 DeepSeek Flash 对四类用户问题做简单意图分类。"""

import os

import json5
from openai import OpenAI


SYSTEM_PROMPT = """你是意图分类器。判断用户输入属于以下四类中的哪一类：
- chat：闲聊、问候、日常交流，不要求专业解答。
- medical：疾病、症状、药物、检查等医疗问题。
- computer：编程、软件、硬件、网络等计算机问题。
- finance：投资、理财、银行、税务等金融问题。

只输出一个 JSON 对象，固定包含以下三个字符串字段，不要输出解释或 Markdown：
{"intent_type": "", "chat_content": "", "intent_query": ""}

规则：
1. intent_type 填 chat、medical、computer 或 finance。
2. chat_content 仅在 chat 时填写一句简短、自然的闲聊回复；其他类型填空字符串 ""。
3. intent_query 仅在 medical、computer、finance 时填写用户要解决的具体问题，保留关键条件，不要回答问题；chat 时填空字符串 ""。
4. 不存在的内容一律用空字符串 ""，不要用 null，也不要省略字段。
5. 同时涉及多个领域时，按用户主要想解决的问题选择一个类型。

"""


def classify_intent(client: OpenAI, query: str) -> dict[str, str]:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = response.choices[0].message.content or ""
    result = json5.loads(content)

    expected_keys = {"intent_type", "chat_content", "intent_query"}
    if not isinstance(result, dict) or set(result) != expected_keys:
        raise ValueError(f"模型返回的字段不符合要求：{content}")
    if result["intent_type"] not in {"chat", "medical", "computer", "finance"}:
        raise ValueError(f"模型返回了未知意图：{content}")
    if not all(isinstance(value, str) for value in result.values()):
        raise ValueError(f"模型返回的字段必须都是字符串：{content}")
    return result


def main() -> None:
    queries = [
        "你好，今天过得怎么样？",
        "我这两天一直咳嗽，应该去看什么科？",
        "Python 怎么读取一个 JSON 文件？",
        "定期存款和货币基金有什么区别？",
    ]

    with OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        timeout=30,
    ) as client:
        for query in queries:
            result = classify_intent(client, query)
            print(f"query: {query}")
            print(f"result: {result}\n")


if __name__ == "__main__":
    main()
