"""教学用模型包装：每次模型调用保存一份可读的 JSON。"""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from agentscope.model import OpenAIChatModel
from paper_trail.runtime import ROOT, llm_response_view


class RecordingModel(OpenAIChatModel):
    def __init__(self, *, directory: Path | None = None, **kwargs):
        super().__init__(**kwargs)
        # 每次运行单独建目录，避免覆盖以前的演示记录。
        self.directory = directory or Path("data/paper_trail/simple") / uuid4().hex
        self.directory = Path(self.directory)
        if not self.directory.is_absolute():
            self.directory = ROOT / self.directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.turn_index = 0
        self.call_index = 0
        self._secrets = tuple(
            value for value in (
                kwargs.get("api_key"), os.getenv("HF_TOKEN"),
                os.getenv("DEEPSEEK_API_KEY"), os.getenv("LLM_API_KEY"),
            ) if isinstance(value, str) and value
        )

    def begin_turn(self):
        """用户每输入一句话，开始新的一轮；一轮可能调用模型多次。"""
        self.turn_index += 1

    def save_json(self, path: Path, record: dict):
        # 中文直接显示；先替换密钥再序列化，保证特殊字符不会破坏 JSON。
        text = json.dumps(record, ensure_ascii=False)
        for secret in self._secrets:
            text = text.replace(json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]")
        text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    async def __call__(self, messages, tools=None, tool_choice=None,
                       structured_model=None, **kwargs):
        self.call_index += 1
        # UUID 也允许显式复用目录而不会覆盖旧记录。
        path = self.directory / (
            f"turn_{self.turn_index:03d}_call_{self.call_index:03d}_{uuid4().hex}.json"
        )
        started = time.monotonic()
        record = {
            "turn_index": self.turn_index,
            "call_index": self.call_index,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model_name,
            "status": "running",
            "request": {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
                "tool_choice": tool_choice,
            },
            "response": None,
            "usage": None,
        }
        self.save_json(path, record)

        def capture(chunk):
            # AgentScope 的流式 chunk 是累计快照，直接取最后一份即可。
            record["response"] = llm_response_view(chunk)
            if chunk.usage is not None:
                record["usage"] = {
                    "input_tokens": chunk.usage.input_tokens,
                    "output_tokens": chunk.usage.output_tokens,
                    "total_tokens": chunk.usage.input_tokens + chunk.usage.output_tokens,
                }

        def finish(status):
            record["status"] = status
            record["elapsed_seconds"] = round(time.monotonic() - started, 2)
            self.save_json(path, record)

        try:
            response = await super().__call__(
                messages, tools=tools, tool_choice=tool_choice,
                structured_model=structured_model, **kwargs,
            )
        except BaseException as exc:
            record["error_type"] = type(exc).__name__
            finish("failed")
            raise

        if not self.stream:
            capture(response)
            finish("completed")
            return response

        async def stream():
            status = "interrupted"
            try:
                async for chunk in response:
                    capture(chunk)
                    yield chunk
                status = "completed"
            except BaseException as exc:
                record["error_type"] = type(exc).__name__
                status = "failed"
                raise
            finally:
                try:
                    await response.aclose()
                finally:
                    finish(status)

        return stream()
