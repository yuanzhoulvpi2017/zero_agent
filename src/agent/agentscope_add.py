from agentscope.formatter import OpenAIChatFormatter


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
