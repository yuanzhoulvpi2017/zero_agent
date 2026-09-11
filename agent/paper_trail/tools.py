"""Read-only Python tools rewritten from the supplied huggingface-papers skill."""

import asyncio
import json
import re
from datetime import date
from urllib.parse import urlparse

import httpx
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
from .cache import ResponseCache
from .compact import compact_result


def paper_id(value: str) -> str:
    value = value.strip()
    if "://" in value:
        url = urlparse(value)
        if url.scheme != "https" or url.hostname not in {
            "huggingface.co",
            "hf.co",
            "arxiv.org",
            "www.arxiv.org",
        }:
            raise ValueError("仅接受 Hugging Face 或 arXiv 的 HTTPS 论文链接。")
        value = re.sub(r"^/(papers|abs|pdf|html)/", "", url.path)
    value = re.sub(r"\.(md|pdf)$", "", value)
    if not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", value):
        raise ValueError("请输入现代 arXiv ID，例如 2602.08025，或对应论文链接。")
    return value


def response(data: dict) -> ToolResponse:
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            )
        ]
    )


class PaperTools:
    def __init__(
        self,
        client: httpx.AsyncClient,
        cache: ResponseCache | None = None,
        token: str = "",
        feed_ttl: int = 900,
        paper_ttl: int = 86400,
    ):
        self.raw_keys = set()
        self.client = client
        self.cache = cache
        self.token = token
        self.feed_ttl = feed_ttl
        self.paper_ttl = paper_ttl

    async def fetch(
        self, path: str, params: dict | None = None, markdown: bool = False
    ) -> dict:
        if self.cache:
            ttl = (
                self.feed_ttl
                if path in {"/api/papers/search", "/api/daily_papers"}
                else self.paper_ttl
            )
            key = self.cache.key(path, params, markdown, self.token)
            result = await self.cache.get_or_fetch(
                key, ttl, lambda: self._fetch(path, params, markdown)
            )
        else:
            result = await self._fetch(path, params, markdown)
        if self.cache and result.get("ok"):
            self.raw_keys.add(key)
            result = {**result, "raw_ref": key}
        return result if markdown else compact_result(path, result)

    async def _fetch(self, path: str, params: dict | None, markdown: bool) -> dict:
        url = "https://huggingface.co" + path
        for attempt in range(3):
            try:
                r = await self.client.get(url, params=params)
                if r.status_code in (429, 502, 503, 504) and attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                if r.status_code == 404:
                    return {
                        "ok": False,
                        "source": url,
                        "error": "论文尚未索引或资源不存在；可检查 arXiv 原文。",
                    }
                r.raise_for_status()
                if markdown and "text/html" in r.headers.get("content-type", ""):
                    return {
                        "ok": False,
                        "source": url,
                        "error": "服务器返回 HTML，未获得可读 Markdown。",
                    }
                return {
                    "ok": True,
                    "source": str(r.url),
                    "data": r.text if markdown else r.json(),
                }
            except httpx.HTTPStatusError as exc:
                return {
                    "ok": False,
                    "source": url,
                    "error": f"HTTP {exc.response.status_code}；401/403 请检查 HF_TOKEN，429 请稍后重试。",
                }
            except (httpx.RequestError, ValueError):
                return {
                    "ok": False,
                    "source": url,
                    "error": "网络请求失败或响应格式异常，请稍后重试。",
                }
        return {"ok": False, "source": url, "error": "请求重试耗尽。"}

    async def search_papers(self, query: str, limit: int = 5) -> ToolResponse:
        """Search Hugging Face papers by topic, title or author; refine the query across turns.

        Args:
            query: Search text, 1 to 250 characters. English technical terms often help.
            limit: Maximum results, 1 to 20.
        """
        if not 1 <= len(query.strip()) <= 250 or not 1 <= limit <= 20:
            return response(
                {"ok": False, "error": "query 长度应为 1–250，limit 应为 1–20。"}
            )
        return response(
            await self.fetch("/api/papers/search", {"q": query, "limit": limit})
        )

    async def daily_papers(
        self, day: str = "", limit: int = 5, page: int = 0
    ) -> ToolResponse:
        """Read the Daily Papers feed, which is a curated feed rather than all published papers.

        Args:
            day: Optional calendar date YYYY-MM-DD; empty means current feed.
            limit: Results per page, 1 to 20.
            page: Zero-based page number.
        """
        try:
            if day:
                date.fromisoformat(day)
            if not 1 <= limit <= 20 or page < 0:
                raise ValueError()
        except ValueError:
            return response(
                {"ok": False, "error": "检查日期 YYYY-MM-DD、limit 1–20 和非负 page。"}
            )
        params = {"limit": limit, "p": page, "sort": "publishedAt"}
        if day:
            params["date"] = day
        return response(await self.fetch("/api/daily_papers", params))

    async def paper_metadata(self, paper: str) -> ToolResponse:
        """Get paper abstract, authors, publication date, GitHub and project links.

        Args:
            paper: arXiv ID or a Hugging Face/arXiv paper URL.
        """
        try:
            pid = paper_id(paper)
        except ValueError as exc:
            return response({"ok": False, "error": str(exc)})
        result = await self.fetch("/api/papers/" + re.sub(r"v\d+$", "", pid))
        result["arxiv_url"] = "https://arxiv.org/abs/" + pid
        result["version_note"] = "HF 元数据按无版本 ID 查询，不能保证对应指定修订版。"
        return response(result)

    async def read_paper(
        self, paper: str, offset: int = 0, length: int = 4000
    ) -> ToolResponse:
        """Read a paper in bounded Markdown chunks; continue with next_offset when needed.

        Args:
            paper: arXiv ID or Hugging Face/arXiv paper URL.
            offset: Character offset, starting at zero.
            length: Characters per chunk, 1000 to 8000.
        """
        try:
            pid = paper_id(paper)
            if offset < 0 or not 1000 <= length <= 8000:
                raise ValueError("offset 必须非负，length 应为 1000–8000。")
        except ValueError as exc:
            return response({"ok": False, "error": str(exc)})
        result = await self.fetch("/papers/" + pid + ".md", markdown=True)
        result["arxiv_url"] = "https://arxiv.org/abs/" + pid
        result["pdf_url"] = "https://arxiv.org/pdf/" + pid
        if result["ok"]:
            text = result.pop("data")
            result.update(
                content=text[offset : offset + length],
                total_characters=len(text),
                next_offset=offset + length if offset + length < len(text) else None,
                coverage="HF Markdown；可能仅为论文介绍页，不能默认是全文。请根据内容确认。",
            )
        return response(result)

    async def linked_resources(
        self, paper: str, kind: str = "models", limit: int = 5
    ) -> ToolResponse:
        """Find Hugging Face models, datasets or Spaces linked to a paper.

        Args:
            paper: arXiv ID or Hugging Face/arXiv paper URL.
            kind: One of models, datasets, spaces.
            limit: Maximum results, 1 to 20.
        """
        try:
            pid = re.sub(r"v\d+$", "", paper_id(paper))
            if kind not in {"models", "datasets", "spaces"} or not 1 <= limit <= 20:
                raise ValueError("kind 应为 models/datasets/spaces，limit 应为 1–20。")
        except ValueError as exc:
            return response({"ok": False, "error": str(exc)})
        return response(
            await self.fetch("/api/" + kind, {"filter": "arxiv:" + pid, "limit": limit})
        )
