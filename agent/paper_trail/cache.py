"""Successful GET response cache, shared by sessions and retained across restarts."""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4


class ResponseCache:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.locks: dict[str, asyncio.Lock] = {}

    def key(self, path: str, params: dict | None, markdown: bool, token: str) -> str:
        scope = hashlib.sha256(token.encode()).hexdigest()
        payload = json.dumps([path, params or {}, markdown, scope], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def get_or_fetch(self, key: str, ttl: int, fetch):
        # Coalesce identical concurrent requests within this server process.
        async with self.locks.setdefault(key, asyncio.Lock()):
            file = self.directory / (key + ".json")
            try:
                cached = json.loads(file.read_text())
                if time.time() - cached["stored_at"] < ttl:
                    return {**cached["response"], "cache_hit": True}
            except (OSError, ValueError, KeyError, TypeError):
                pass
            result = await fetch()
            if result.get("ok"):
                temporary = file.with_suffix("." + uuid4().hex + ".tmp")
                try:
                    temporary.write_text(
                        json.dumps(
                            {"stored_at": time.time(), "response": result},
                            ensure_ascii=False,
                        )
                    )
                    os.replace(temporary, file)
                finally:
                    temporary.unlink(missing_ok=True)
            return {**result, "cache_hit": False}
