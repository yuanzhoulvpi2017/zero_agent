import asyncio
import contextlib
import socket
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from src.mcp.simple_tool import server as simple_tool_server
import logging

_LOCAL_IP = socket.gethostbyname(socket.gethostname())

logger = logging.getLogger(__name__)


simple_tool_app = simple_tool_server.streamable_http_app()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(simple_tool_server.session_manager.run())
            yield
    finally:
        print("Shutting down simple tool server...")


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每条请求的机器IP、客户端IP、方法、路径和耗时"""
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000  # ms
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        "[%s] client=%s method=%s path=%s status=%s duration=%.1fms",
        _LOCAL_IP,
        client_ip,
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


# 挂载mcpserver接口，所有的mcp都要走/mcpserver/{mcp_name}路径
app.mount("/mcpserver/simple_tool/", simple_tool_app)

## 后续请求这个服务的时候，路径是：http://192.168.0.106:8123/mcpserver/simple_tool/mcp


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8123, access_log=False)
