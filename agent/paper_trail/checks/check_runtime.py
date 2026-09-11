import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from agentscope.message import Msg
from agentscope.tool import Toolkit
from paper_trail.cache import ResponseCache
from paper_trail.runtime import (
    DeepSeekFormatter,
    PaperSession,
    conversation_view,
    list_session_summaries,
)
from paper_trail.tools import PaperTools, paper_id


class Checks(unittest.IsolatedAsyncioTestCase):
    def test_conversation_view_and_listing(self):
        messages = [
            {
                "id": "u1",
                "name": "user",
                "role": "user",
                "content": "介绍一下记忆论文",
                "metadata": {},
                "timestamp": "2026-09-09 08:00:00.000",
            },
            {
                "id": "a1",
                "name": "小埋",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "text", "text": "先检索。"},
                    {
                        "type": "tool_use",
                        "id": "call1",
                        "name": "search_papers",
                        "input": {"query": "memory"},
                    },
                ],
            },
            {
                "id": "t1",
                "name": "system",
                "role": "system",
                "content": [
                    {
                        "type": "tool_result",
                        "id": "call1",
                        "name": "search_papers",
                        "output": [
                            {
                                "type": "text",
                                "text": '{"ok":true,"raw_ref":"ab","cache_hit":false}',
                            }
                        ],
                    }
                ],
            },
            {
                "id": "a2",
                "name": "小埋",
                "role": "assistant",
                "content": [{"type": "text", "text": "找到一篇。"}],
            },
        ]
        view = conversation_view(
            {
                "session_id": "a" * 32,
                "title": "",
                "created_at": "2026-09-09T00:00:00+00:00",
                "status": "completed",
                "turn_usage": [
                    {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "model_calls": 2,
                        "complete": True,
                        "elapsed_seconds": 1.2,
                        "calls": [],
                    }
                ],
                "session_usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "model_calls": 2,
                    "complete": True,
                    "elapsed_seconds": 1.2,
                    "turns": 1,
                },
            },
            messages,
        )
        self.assertEqual(view["title"], "介绍一下记忆论文")
        self.assertNotIn("hidden", json.dumps(view, ensure_ascii=False))
        self.assertEqual(
            [item["role"] for item in view["messages"]],
            ["user", "assistant", "assistant"],
        )
        self.assertEqual(view["tools"][0]["state"], "completed")
        self.assertEqual(view["tools"][0]["raw_ref"], "ab")
        self.assertEqual(view["messages"][-1]["usage"]["total_tokens"], 15)
        with tempfile.TemporaryDirectory() as directory:
            session_id = "b" * 32
            path = Path(directory) / "data/paper_trail/web" / session_id
            path.mkdir(parents=True)
            (path / "manifest.json").write_text(
                json.dumps(
                    {
                        "session_id": session_id,
                        "title": "介绍一下记忆论文",
                        "created_at": "2026-09-09T00:00:00+00:00",
                        "updated_at": "2026-09-09T01:00:00+00:00",
                        "status": "completed",
                    }
                )
            )
            (path / "trajectory.json").write_text(json.dumps(messages))
            with patch("paper_trail.runtime.ROOT", Path(directory)):
                listed = list_session_summaries()
            self.assertEqual(listed[0]["session_id"], session_id)
            self.assertEqual(listed[0]["title"], "介绍一下记忆论文")
    async def test_cache_and_failures(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, json={"id": "2602.08025"})

        with tempfile.TemporaryDirectory() as directory:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                cache = ResponseCache(Path(directory))
                tool = PaperTools(client, cache)
                results = await asyncio.gather(
                    *[tool.fetch("/api/papers/x", {"a": 1, "b": 2}) for _ in range(5)]
                )
                self.assertEqual(len(calls), 1)
                self.assertEqual(sum(r["cache_hit"] for r in results), 4)
                tool.cache = ResponseCache(Path(directory))
                await tool.fetch("/api/papers/x", {"b": 2, "a": 1})
                self.assertEqual(len(calls), 1)
                await tool.fetch("/api/papers/x", {"a": 2})
                self.assertEqual(len(calls), 2)
                tool.paper_ttl = 0
                await tool.fetch("/api/papers/x", {"a": 2})
                self.assertEqual(len(calls), 3)
            failed_calls = []

            def fail(request):
                failed_calls.append(request)
                return httpx.Response(404)

            async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
                tool = PaperTools(client, cache)
                for _ in range(2):
                    self.assertFalse((await tool.fetch("/missing"))["ok"])
                self.assertEqual(len(failed_calls), 2)

    async def test_formatter_all_assistant_turns(self):
        messages = [
            Msg(
                "assistant",
                [
                    {"type": "thinking", "thinking": "first"},
                    {"type": "tool_use", "id": "call1", "name": "search", "input": {}},
                ],
                "assistant",
            ),
            Msg(
                "tool",
                [
                    {
                        "type": "tool_result",
                        "id": "call1",
                        "name": "search",
                        "output": [{"type": "text", "text": "ok"}],
                    }
                ],
                "system",
            ),
            Msg(
                "assistant",
                [
                    {"type": "thinking", "thinking": "second"},
                    {"type": "text", "text": "answer"},
                ],
                "assistant",
            ),
        ]
        formatted = await DeepSeekFormatter().format(messages)
        self.assertEqual(
            [m["reasoning_content"] for m in formatted if m["role"] == "assistant"],
            ["first", "second"],
        )
        self.assertEqual(formatted[1]["tool_call_id"], "call1")

    async def test_tools_and_registration(self):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, text="x" * 21000)
            )
        ) as client:
            tools = PaperTools(client)
            toolkit = Toolkit()
            for method in (
                tools.search_papers,
                tools.daily_papers,
                tools.paper_metadata,
                tools.read_paper,
                tools.linked_resources,
            ):
                toolkit.register_tool_function(method)
            data = json.loads(
                (await tools.read_paper("2602.08025", length=1000)).content[0]["text"]
            )
            self.assertEqual(data["next_offset"], 1000)
            self.assertEqual(len(data["content"]), 1000)
            bad = json.loads((await tools.daily_papers("bad")).content[0]["text"])
            self.assertFalse(bad["ok"])
        self.assertEqual(
            paper_id("https://arxiv.org/pdf/2602.08025v2.pdf"), "2602.08025v2"
        )
        with self.assertRaises(ValueError):
            paper_id("https://evil.example/2602.08025")

    async def test_agent_tool_cycle_and_trace(self):
        import tomllib
        from paper_trail.runtime import ROOT

        config = tomllib.loads((ROOT / "configs/paper_trail/agent.toml").read_text())
        requests = []

        def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            if len(requests) == 1:
                message = {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "search plan",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "search_papers",
                                "arguments": '{"query":"memory"}',
                            },
                        }
                    ],
                }
            else:
                message = {
                    "role": "assistant",
                    "content": "Found a paper",
                    "reasoning_content": "summary plan",
                }
            self.assertTrue(payload["stream"])
            deltas = [
                {"role": "assistant", "reasoning_content": message["reasoning_content"]}
            ]
            if message.get("tool_calls"):
                tool_call = {**message["tool_calls"][0], "index": 0}
                deltas.append({"tool_calls": [tool_call]})
            else:
                deltas.extend([{"content": "Found "}, {"content": "a paper"}])
            chunks = []
            for delta in deltas:
                chunk = {
                    "id": "mock",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "mock",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                chunks.append("data: " + json.dumps(chunk) + "\n\n")
            chunks.append(
                "data: "
                + json.dumps(
                    {
                        "id": "mock",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "mock",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                )
                + "\n\n"
            )
            chunks.append("data: [DONE]\n\n")
            return httpx.Response(
                200, text="".join(chunks), headers={"content-type": "text/event-stream"}
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-secret"}),
        ):
            with patch("paper_trail.runtime.ROOT", Path(directory)):
                session = PaperSession(config, ResponseCache(Path(directory) / "cache"))
            import openai

            session.agent.model.client = openai.AsyncOpenAI(
                api_key="test-secret",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )

            async def fake_fetch(*args, **kwargs):
                return {"ok": True, "data": [{"id": "2602.08025"}]}

            with patch.object(PaperTools, "fetch", fake_fetch):
                events = [
                    event async for event in session.stream_chat("Find memory papers")
                ]
                self.assertEqual(events[-1]["answer"], "Found a paper")
                self.assertEqual(events[-1]["usage"]["input_tokens"], 20)
                self.assertEqual(events[-1]["usage"]["output_tokens"], 10)
                self.assertEqual(events[-1]["usage"]["total_tokens"], 30)
                self.assertEqual(events[-1]["usage"]["model_calls"], 2)
                self.assertTrue(any(event.get("text") == "Found " for event in events))
                self.assertTrue(
                    any("search_papers" in event.get("status", "") for event in events)
                )
                self.assertEqual(session.status["state"], "completed")
                tool_events = [
                    tool for event in events for tool in event.get("tools", [])
                ]
                self.assertTrue(
                    any(
                        t["name"] == "search_papers"
                        and t.get("input") == {"query": "memory"}
                        for t in tool_events
                    )
                )
                self.assertTrue(
                    any(
                        t["state"] == "completed"
                        and "2602.08025" in t.get("output", "")
                        for t in tool_events
                    )
                )
                second = [event async for event in session.stream_chat("Compare them")]
                self.assertEqual(second[-1]["usage"]["total_tokens"], 15)
                self.assertEqual(second[-1]["session_usage"]["input_tokens"], 30)
                self.assertEqual(second[-1]["session_usage"]["output_tokens"], 15)
                self.assertEqual(second[-1]["session_usage"]["total_tokens"], 45)
                self.assertEqual(second[-1]["session_usage"]["model_calls"], 3)
                self.assertEqual(second[-1]["session_usage"]["turns"], 2)
                self.assertEqual(events[-1]["session_usage"]["total_tokens"], 30)
            self.assertEqual(
                requests[1]["messages"][-2]["reasoning_content"], "search plan"
            )
            self.assertEqual(
                requests[0]["extra_body"]
                if "extra_body" in requests[0]
                else requests[0]["thinking"],
                {"type": "enabled"},
            )
            trace = json.loads((session.directory / "trajectory.json").read_text())
            self.assertTrue(
                any(
                    b.get("type") == "tool_result"
                    for m in trace
                    for b in m["content"]
                    if isinstance(b, dict)
                )
            )
            conversation = json.loads(
                (session.directory / "conversation.json").read_text()
            )
            self.assertEqual(conversation["title"], "Find memory papers")
            self.assertEqual(
                [item["text"] for item in conversation["messages"] if item["role"] == "user"],
                ["Find memory papers", "Compare them"],
            )
            self.assertTrue(any(item["name"] == "search_papers" for item in conversation["tools"]))
            llm_calls = json.loads((session.directory / "llm_calls.json").read_text())
            self.assertGreaterEqual(len(llm_calls), 2)
            self.assertIn("messages", llm_calls[0]["request"])
            self.assertEqual(llm_calls[0]["turn_index"], 0)
            chain = json.loads((session.directory / "dialogue_chain.json").read_text())
            self.assertEqual(chain["turn_count"], 2)
            self.assertEqual(chain["turns"][0]["user"]["text"], "Find memory papers")
            self.assertTrue(chain["turns"][0]["llm_calls"])
            with patch("paper_trail.runtime.ROOT", Path(directory)):
                summaries = list_session_summaries()
                self.assertEqual(summaries[0]["session_id"], session.id)
                restored = await PaperSession.restore(
                    session.id, config, ResponseCache(Path(directory) / "cache")
                )
                memory = await restored.agent.memory.get_memory()
                self.assertEqual(len(memory), len(trace))
                self.assertEqual(len(restored.agent.model.llm_calls), len(llm_calls))
                await restored.close()
            await session.agent.model.client.close()
            await session.close()


if __name__ == "__main__":
    unittest.main()
