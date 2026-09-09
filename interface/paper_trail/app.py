"""Local browser interface. Run a single worker to preserve in-memory sessions."""

import asyncio
import json
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from markdown_it import MarkdownIt
from paper_trail.cache import ResponseCache
from paper_trail.runtime import (
    SESSION_ID,
    PaperSession,
    data_root,
    find_session_directory,
    list_session_summaries,
    read_session_record,
    settings,
)

markdown = (
    MarkdownIt("commonmark", {"html": False}).enable("table").enable("strikethrough")
)
INDEX = Path(__file__).with_name("index.html")
SPA_PAGES = {
    "chat",
    "history",
    "traces",
    "dataset",
    "training",
    "evaluation",
}
SPA_DETAIL = {"chat", "history", "traces"}
RAW_REF = re.compile(r"^[0-9a-f]{64}$")
LIVE_LIMIT = 50
sessions = {}
locks = {}


@asynccontextmanager
async def lifespan(app):
    app.state.cache = ResponseCache(data_root() / "cache")
    yield
    for session in sessions.values():
        await session.close()
    sessions.clear()
    locks.clear()


app = FastAPI(title="PaperTrail", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=12000)


def decorate(conversation: dict) -> dict:
    data = {**conversation, "messages": [dict(item) for item in conversation["messages"]]}
    for item in data["messages"]:
        if item.get("role") == "assistant" and item.get("text"):
            item["html"] = markdown.render(item["text"])
    return data


async def evict_idle():
    if len(sessions) < LIVE_LIMIT:
        return
    for session_id, lock in list(locks.items()):
        if lock.locked():
            continue
        session = sessions.pop(session_id, None)
        locks.pop(session_id, None)
        if session:
            await session.close()
        if len(sessions) < LIVE_LIMIT:
            return
    raise HTTPException(503, "本地会话数已达上限，请等待进行中的回复完成。")


async def live_session(session_id: str | None):
    if session_id:
        session = sessions.get(session_id)
        if session:
            return session
        try:
            session = await PaperSession.restore(
                session_id, settings(), app.state.cache
            )
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
        if not session:
            raise HTTPException(404, "会话不存在，请到「历史对话」打开已保存记录。")
        sessions[session.id] = session
        locks[session.id] = asyncio.Lock()
        return session
    await evict_idle()
    try:
        session = PaperSession(settings(), app.state.cache)
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc
    sessions[session.id] = session
    locks[session.id] = asyncio.Lock()
    return session


@app.get("/")
async def index():
    return FileResponse(INDEX)


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": list_session_summaries()}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    record = read_session_record(session_id)
    if not record:
        raise HTTPException(404, "未找到该对话记录。")
    data = decorate(record["conversation"])
    data["busy"] = session_id in locks and locks[session_id].locked()
    data["live"] = session_id in sessions
    return data


@app.post("/api/chat")
async def chat(body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(422, "消息不能为空。")
    session = await live_session(body.session_id)
    lock = locks[session.id]
    if lock.locked():
        raise HTTPException(409, "该会话正在处理上一条消息。")
    await lock.acquire()

    async def events():
        completed = False
        stream = session.stream_chat(body.message)
        try:
            yield (
                "data: "
                + json.dumps({"type": "session", "session_id": session.id})
                + "\n\n"
            )
            async for event in stream:
                if event["type"] in {"done", "error"}:
                    completed = True
                if event.get("text"):
                    event = {**event, "html": markdown.render(event["text"])}
                if event["type"] == "done":
                    event = {
                        **event,
                        "html": markdown.render(event.get("answer") or ""),
                    }
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        finally:
            await stream.aclose()
            try:
                if not completed:
                    sessions.pop(session.id, None)
                    locks.pop(session.id, None)
                    await session.close()
            finally:
                lock.release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sessions/{session_id}/raw/{raw_ref}")
async def raw_tool_response(session_id: str, raw_ref: str):
    session = sessions.get(session_id)
    if session:
        if raw_ref not in session.paper_tools.raw_keys:
            raise HTTPException(404, "该会话没有这份原始结果，请重新查询。")
        try:
            cached = json.loads(
                (session.paper_tools.cache.directory / (raw_ref + ".json")).read_text()
            )
            return cached["response"]
        except (OSError, ValueError, KeyError):
            raise HTTPException(404, "原始缓存已清理，请重新查询。")
    if not SESSION_ID.fullmatch(session_id) or not RAW_REF.fullmatch(raw_ref):
        raise HTTPException(404, "该会话没有这份原始结果，请重新查询。")
    if not read_session_record(session_id):
        raise HTTPException(404, "未找到该对话记录。")
    try:
        cached = json.loads(
            (data_root() / "cache" / (raw_ref + ".json")).read_text()
        )
        return cached["response"]
    except (OSError, ValueError, KeyError):
        raise HTTPException(404, "原始缓存已清理，请重新查询。")


@app.get("/api/sessions/{session_id}/status")
async def session_status(session_id: str):
    session = sessions.get(session_id)
    if session:
        return {
            "session_id": session.id,
            **session.status,
            "busy": locks[session_id].locked(),
        }
    record = read_session_record(session_id)
    if not record:
        raise HTTPException(404, "会话不存在或已释放。")
    status = record["manifest"].get("status") or "unknown"
    labels = {
        "created": "已保存，等待继续",
        "running": "上次未完成，可继续追问",
        "completed": "已保存",
        "failed": "上次失败，记录仍在",
    }
    return {
        "session_id": session_id,
        "state": status,
        "stage": labels.get(status, status),
        "busy": False,
    }


@app.get("/api/sessions/{session_id}/trace")
async def session_trace(session_id: str):
    record = read_session_record(session_id)
    if not record:
        raise HTTPException(404, "未找到该对话记录。")
    conversation = record["conversation"]
    manifest = record["manifest"]
    return {
        "session_id": session_id,
        "title": conversation["title"],
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
        "status": conversation["status"],
        "session_usage": conversation["session_usage"],
        "code_commit": manifest.get("code_commit"),
        "config": manifest.get("config"),
        "trajectory": record["trajectory"],
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, purge: bool = False):
    session = sessions.get(session_id)
    if session:
        if locks[session_id].locked():
            raise HTTPException(409, "请等待当前回复完成。")
        await session.close()
        sessions.pop(session_id)
        locks.pop(session_id)
    if purge:
        if not SESSION_ID.fullmatch(session_id):
            raise HTTPException(404, "未找到该对话记录。")
        directory = find_session_directory(session_id)
        if directory and directory.is_dir():
            shutil.rmtree(directory)
    return {"ok": True}


@app.get("/{page}")
async def spa_page(page: str):
    if page not in SPA_PAGES:
        raise HTTPException(404, "页面不存在。")
    return FileResponse(INDEX)


@app.get("/{page}/{session_id}")
async def spa_detail(page: str, session_id: str):
    if page not in SPA_DETAIL or not SESSION_ID.fullmatch(session_id):
        raise HTTPException(404, "页面不存在。")
    return FileResponse(INDEX)
