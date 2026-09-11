"""Run from repo root: interface/.venv/bin/python interface/paper_trail/check_web.py."""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import app


class WebChecks(unittest.TestCase):
    def test_page_and_errors(self):
        with TestClient(app) as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/chat").status_code, 200)
            self.assertEqual(client.get("/history").status_code, 200)
            self.assertEqual(client.get("/traces").status_code, 200)
            self.assertEqual(client.get("/dataset").status_code, 200)
            self.assertEqual(client.get("/training").status_code, 200)
            self.assertEqual(client.get("/evaluation").status_code, 200)
            self.assertEqual(client.get("/not-a-page").status_code, 404)
            self.assertEqual(
                client.post("/api/chat", json={"message": " "}).status_code, 422
            )
            self.assertEqual(
                client.post(
                    "/api/chat", json={"message": "hi", "session_id": "missing"}
                ).status_code,
                404,
            )
            with patch("app.settings", side_effect=ValueError("missing key")):
                self.assertEqual(
                    client.post("/api/chat", json={"message": "hi"}).status_code, 503
                )

    def test_stream_and_status(self):
        class FakeSession:
            id = "stream-check"
            status = {"state": "idle", "stage": "等待输入"}
            closed = False

            def __init__(self, *args):
                pass

            async def stream_chat(self, text):
                self.status = {"state": "running", "stage": "正在生成回复"}
                yield {
                    "type": "message",
                    "message_id": "m",
                    "text": "你",
                    "status": "正在生成回复",
                }
                yield {
                    "type": "message",
                    "message_id": "m",
                    "text": "你好",
                    "status": "正在生成回复",
                }
                self.status = {"state": "completed", "stage": "回复完成"}
                yield {"type": "done", "answer": "你好"}

            async def close(self):
                self.closed = True

        import json

        with (
            TestClient(app) as client,
            patch("app.PaperSession", FakeSession),
            patch("app.settings", return_value={}),
        ):
            result = client.post("/api/chat", json={"message": "hello"})
            self.assertIn("text/event-stream", result.headers["content-type"])
            events = [
                json.loads(frame[6:]) for frame in result.text.strip().split("\n\n")
            ]
            self.assertEqual(
                [e["type"] for e in events], ["session", "message", "message", "done"]
            )
            status = client.get("/api/sessions/stream-check/status").json()
            self.assertEqual(status["state"], "completed")
            self.assertFalse(status["busy"])
            self.assertEqual(
                client.delete("/api/sessions/stream-check").status_code, 200
            )
            self.assertEqual(
                client.get("/api/sessions/stream-check/status").status_code, 404
            )

    def test_stream_error_keeps_saved_session(self):
        class FailedSession:
            id = "failed-check"
            status = {"state": "failed", "stage": "执行失败或连接中断"}

            def __init__(self, *args, **kwargs):
                pass

            async def stream_chat(self, text):
                yield {"type": "error", "message": "模拟失败"}

            async def close(self):
                pass

        with (
            TestClient(app) as client,
            patch("app.PaperSession", FailedSession),
            patch("app.settings", return_value={}),
        ):
            result = client.post("/api/chat", json={"message": "hello"})
            self.assertIn("模拟失败", result.text)
            self.assertEqual(
                client.get("/api/sessions/failed-check/status").status_code, 200
            )

    def test_raw_response_access(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        import app as server

        with tempfile.TemporaryDirectory() as directory, TestClient(app) as client:
            raw = {"ok": True, "data": {"summary": "x" * 9000}}
            Path(directory, "known.json").write_text(json.dumps({"response": raw}))
            server.sessions["raw-check"] = SimpleNamespace(
                paper_tools=SimpleNamespace(
                    raw_keys={"known"}, cache=SimpleNamespace(directory=Path(directory))
                )
            )
            try:
                self.assertEqual(
                    client.get("/api/sessions/raw-check/raw/known").json(), raw
                )
                self.assertEqual(
                    client.get("/api/sessions/raw-check/raw/unknown").status_code, 404
                )
                self.assertEqual(
                    client.get("/api/sessions/other/raw/known").status_code, 404
                )
            finally:
                server.sessions.pop("raw-check")

    def test_markdown_rendering(self):
        from app import markdown

        html = markdown.render(
            "# Title\n\n**bold** [paper](https://huggingface.co/papers/2602.08025)\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```python\nprint(1)\n```\n<script>alert(1)</script>\n\n[x](javascript:alert(1))"
        )
        self.assertIn("<h1>Title</h1>", html)
        self.assertIn("<strong>bold</strong>", html)
        self.assertIn("<table>", html)
        self.assertIn('class="language-python"', html)
        self.assertNotIn("<script>", html)
        self.assertNotIn('href="javascript:', html)

    def test_disconnect_releases_session(self):
        import asyncio
        import app as server

        class InterruptedSession:
            id = "interrupted-check"
            closed = False
            stopped = False

            async def stream_chat(self, text):
                try:
                    yield {"type": "message", "message_id": "m", "text": "partial"}
                    await asyncio.Event().wait()
                finally:
                    self.stopped = True

            async def close(self):
                self.closed = True

        async def check():
            session = InterruptedSession()
            lock = asyncio.Lock()
            server.sessions[session.id] = session
            server.locks[session.id] = lock
            response = await server.chat(
                server.ChatRequest(message="hi", session_id=session.id)
            )
            stream = response.body_iterator
            await anext(stream)
            await anext(stream)
            await stream.aclose()
            self.assertTrue(session.closed and session.stopped)
            self.assertFalse(lock.locked())
            self.assertNotIn(session.id, server.sessions)

        asyncio.run(check())

    def test_saved_history_and_restore(self):
        import json
        import tempfile
        from pathlib import Path
        import app as server

        session_id = "a" * 32
        restored = []

        class RestoredSession:
            id = session_id
            status = {"state": "completed", "stage": "回复完成"}
            closed = False

            def __init__(self, *args, **kwargs):
                pass

            @classmethod
            async def restore(cls, ident, config, cache):
                restored.append(ident)
                instance = cls()
                return instance

            async def stream_chat(self, text):
                yield {"type": "done", "answer": "继续"}

            async def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data/paper_trail/web" / session_id
            path.mkdir(parents=True)
            (path / "manifest.json").write_text(
                json.dumps(
                    {
                        "session_id": session_id,
                        "title": "介绍记忆论文",
                        "created_at": "2026-09-09T00:00:00+00:00",
                        "updated_at": "2026-09-09T01:00:00+00:00",
                        "status": "completed",
                        "session_usage": {"turns": 1},
                    }
                )
            )
            (path / "trajectory.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "u1",
                            "name": "user",
                            "role": "user",
                            "content": "介绍记忆论文",
                            "metadata": {},
                            "timestamp": "2026-09-09 08:00:00.000",
                        },
                        {
                            "id": "a1",
                            "name": "小埋",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "先看这几篇。"}],
                        },
                    ]
                )
            )
            with (
                patch("paper_trail.runtime.ROOT", root),
                patch("app.PaperSession", RestoredSession),
                patch("app.settings", return_value={}),
                TestClient(app) as client,
            ):
                listed = client.get("/api/sessions").json()["sessions"]
                self.assertEqual(listed[0]["title"], "介绍记忆论文")
                detail = client.get("/api/sessions/" + session_id).json()
                self.assertIn("<p>先看这几篇。</p>", detail["messages"][1]["html"])
                self.assertEqual(
                    client.get("/history/" + session_id).status_code, 200
                )
                self.assertEqual(
                    client.get("/traces/" + session_id).status_code, 200
                )
                trace = client.get("/api/sessions/" + session_id + "/trace").json()
                self.assertEqual(trace["trajectory"][0]["content"], "介绍记忆论文")
                self.assertEqual(
                    client.get("/api/sessions/" + ("c" * 32) + "/trace").status_code,
                    404,
                )
                result = client.post(
                    "/api/chat",
                    json={"session_id": session_id, "message": "继续比较"},
                )
                self.assertEqual(result.status_code, 200)
                self.assertEqual(restored, [session_id])
                self.assertEqual(
                    client.delete(
                        "/api/sessions/" + session_id + "?purge=true"
                    ).status_code,
                    200,
                )
                self.assertEqual(
                    client.get("/api/sessions/" + session_id).status_code, 404
                )
                self.assertFalse((path / "manifest.json").exists())
            server.sessions.clear()
            server.locks.clear()


if __name__ == "__main__":
    unittest.main()
