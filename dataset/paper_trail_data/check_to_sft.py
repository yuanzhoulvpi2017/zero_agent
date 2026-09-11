"""Sanity checks for trajectory -> SFT conversion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_trail_data.to_sft import (
    assistant_from_response,
    call_to_sample,
    normalize_message,
    normalize_tool_calls,
)


class ToSftTests(unittest.TestCase):
    def test_normalize_tool_calls_parses_json_arguments(self):
        calls = normalize_tool_calls(
            [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "daily_papers",
                        "arguments": '{"limit": 5}',
                    },
                }
            ]
        )
        self.assertEqual(calls[0]["function"]["arguments"], {"limit": 5})

    def test_strips_thinking_by_default(self):
        message = normalize_message(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "secret plan"},
                    {"type": "text", "text": "先查一下"},
                ],
                "reasoning_content": "should drop",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "daily_papers", "arguments": "{}"},
                    }
                ],
            },
            include_thinking=False,
        )
        self.assertEqual(message["content"], "先查一下")
        self.assertNotIn("reasoning_content", message)

        target = assistant_from_response(
            {
                "text": "好的",
                "thinking": "internal",
                "reason": "also internal",
                "tool_calls": [],
            },
            include_thinking=False,
        )
        self.assertEqual(target["content"], "好的")
        self.assertNotIn("reasoning_content", target)

    def test_call_to_sample_appends_assistant_target(self):
        call = {
            "call_index": 0,
            "turn_index": 0,
            "model": "deepseek-v4-flash",
            "request": {
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "daily_papers",
                            "description": "d",
                            "parameters": {
                                "type": "object",
                                "properties": {"limit": {"type": "integer"}},
                            },
                        },
                    }
                ],
            },
            "response": {
                "text": "先查一下",
                "thinking": "需要工具",
                "tool_calls": [{"id": "c1", "name": "daily_papers", "input": {"limit": 3}}],
            },
        }
        sample = call_to_sample(
            call,
            session_id="a" * 32,
            source="x",
            include_thinking=False,
        )
        self.assertIsNotNone(sample)
        self.assertEqual(sample["messages"][-1]["role"], "assistant")
        self.assertNotIn("reasoning_content", sample["messages"][-1])
        self.assertEqual(sample["chat_template_kwargs"]["enable_thinking"], False)
        self.assertEqual(
            sample["messages"][-1]["tool_calls"][0]["function"]["name"], "daily_papers"
        )


if __name__ == "__main__":
    unittest.main()
