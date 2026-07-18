"""MCP 认证测试 — API Key 守卫 (401/200) + 配置 fail-fast."""

import httpx
import pytest
from pydantic import ValidationError

from src.mcp_server.auth import MCPAPIKeyGuard

API_KEY = "test-mcp-key-16chars-min"


async def _ok_app(scope, receive, send):
    """最小 ASGI 应用 — 守卫放行后返回 200."""
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [(b"content-type", b"application/json")],
    })
    await send({"type": "http.response.body", "body": b'{"ok":true}'})


@pytest.fixture
def client():
    guard = MCPAPIKeyGuard(_ok_app, API_KEY)
    transport = httpx.ASGITransport(app=guard)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def test_missing_key_rejected(client):
    async with client:
        resp = await client.post("/mcp")
    assert resp.status_code == 401
    assert "unauthorized" in resp.text
    assert resp.headers.get("www-authenticate") == "Bearer"


async def test_wrong_key_rejected(client):
    async with client:
        resp = await client.post("/mcp", headers={"Authorization": "Bearer wrong-key-000000000"})
    assert resp.status_code == 401


async def test_non_bearer_authorization_rejected(client):
    async with client:
        resp = await client.post("/mcp", headers={"Authorization": f"Basic {API_KEY}"})
    assert resp.status_code == 401


async def test_bearer_key_accepted(client):
    async with client:
        resp = await client.post("/mcp", headers={"Authorization": f"Bearer {API_KEY}"})
    assert resp.status_code == 200


async def test_x_api_key_accepted(client):
    async with client:
        resp = await client.post("/mcp", headers={"X-API-Key": API_KEY})
    assert resp.status_code == 200


# ── 配置 fail-fast (spec mcp-auth) ───────────────────────────


def _make_settings(**kwargs):
    from src.config import Settings

    # _env_file=None 隔离仓库 .env，聚焦被测字段
    return Settings(_env_file=None, **kwargs)


def test_mcp_enabled_without_key_fails_startup():
    with pytest.raises(ValidationError, match="RAG_MCP_API_KEY"):
        _make_settings(RAG_MCP_ENABLED=True)


def test_mcp_enabled_with_short_key_fails_startup():
    with pytest.raises(ValidationError, match="16 characters"):
        _make_settings(RAG_MCP_ENABLED=True, RAG_MCP_API_KEY="too-short")


def test_mcp_enabled_with_strong_key_passes():
    s = _make_settings(RAG_MCP_ENABLED=True, RAG_MCP_API_KEY="x" * 16)
    assert s.mcp_enabled is True


def test_mcp_disabled_needs_no_key():
    s = _make_settings()
    assert s.mcp_enabled is False
    assert s.mcp_api_key == ""
