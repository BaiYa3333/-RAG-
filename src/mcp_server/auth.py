"""MCP HTTP 传输认证 — 静态 API Key ASGI 守卫.

包裹 MCP Streamable HTTP 子应用，在请求进入任何 MCP handler 前校验
``Authorization: Bearer <key>`` 或 ``X-API-Key`` 头（常量时间比较）。

stdio 传输不经过此守卫 — 本地信任模型：进程由用户自己的 Host 以子进程
拉起，操作系统进程边界即信任边界。
"""

from __future__ import annotations

import secrets
from typing import Any

_UNAUTHORIZED_BODY = (
    b'{"error":"unauthorized","detail":"Missing or invalid MCP API key. '
    b'Provide it via \'Authorization: Bearer <key>\' or \'X-API-Key\' header."}'
)


class MCPAPIKeyGuard:
    """ASGI 中间件：/mcp 子应用的 API Key 前置校验。"""

    def __init__(self, app: Any, api_key: str):
        self._app = app
        self._api_key = api_key.encode("utf-8")

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            # lifespan / websocket 等非 HTTP scope 直接透传
            await self._app(scope, receive, send)
            return

        provided = self._extract_key(scope.get("headers") or [])
        if provided is None or not secrets.compare_digest(
            provided.encode("utf-8"), self._api_key
        ):
            await self._reject(send)
            return

        await self._app(scope, receive, send)

    @staticmethod
    def _extract_key(headers: list[tuple[bytes, bytes]]) -> str | None:
        """提取 API Key — X-API-Key 优先，其次 Authorization: Bearer。"""
        auth_header = None
        x_api_key = None
        for name, value in headers:
            lname = name.decode("latin-1").lower()
            if lname == "x-api-key":
                x_api_key = value.decode("latin-1")
            elif lname == "authorization":
                auth_header = value.decode("latin-1")

        if x_api_key:
            return x_api_key.strip()
        if auth_header and auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()
        return None

    @staticmethod
    async def _reject(send: Any) -> None:
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b"Bearer"),
            ],
        })
        await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})
