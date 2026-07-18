"""MCP Streamable HTTP 挂载测试 — /mcp 路径、initialize 握手、disabled 404.

使用独立 FastMCP 实例（create_mcp_server 工厂）：session_manager 每实例
只能 run() 一次，测试间不可复用生产单例。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.mcp_server.auth import MCPAPIKeyGuard
from src.mcp_server.server import create_mcp_server

API_KEY = "test-mcp-key-16chars-min"

INIT_PAYLOAD = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "pytest-client", "version": "0.0.1"},
    },
}
MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _build_enabled_app(api_key: str = API_KEY) -> FastAPI:
    """镜像 main.py 的注册方式：精确 Route("/mcp") + guard + lifespan 托管.

    不用 app.mount：Mount 对不带尾斜杠的 /mcp 会 307 重定向，严格客户端失败。
    """
    from starlette.routing import Route

    mcp = create_mcp_server()
    http_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():
            yield

    app = FastAPI(lifespan=lifespan)
    app.router.routes.append(Route("/mcp", endpoint=MCPAPIKeyGuard(http_app, api_key)))
    return app


def test_initialize_handshake_at_mcp_path():
    """精确路径 /mcp 直接响应（无 307 重定向），有效 Key 完成 initialize 握手."""
    app = _build_enabled_app()
    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json=INIT_PAYLOAD,
            headers={**MCP_HEADERS, "Authorization": f"Bearer {API_KEY}"},
            follow_redirects=False,  # 严格客户端不跟随重定向 — 钉死 Mount 307 回归
        )
    assert resp.status_code == 200
    assert "serverInfo" in resp.text


def test_mcp_path_requires_api_key():
    app = _build_enabled_app()
    with TestClient(app) as client:
        resp = client.post(
            "/mcp", json=INIT_PAYLOAD, headers=MCP_HEADERS, follow_redirects=False
        )
    assert resp.status_code == 401


def test_disabled_by_default_returns_404():
    """RAG_MCP_ENABLED 默认 false → main.app 不挂载 /mcp → 404."""
    from src.config import settings
    from src.main import app

    assert settings.mcp_enabled is False
    client = TestClient(app)
    resp = client.post("/mcp", json=INIT_PAYLOAD, headers=MCP_HEADERS)
    assert resp.status_code == 404
