"""Redis 滑动窗口速率限制中间件 — per-IP 限流，fail-open."""

from __future__ import annotations

import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config import settings
from src.utils.logger import logger

RATE_LIMIT_WINDOW = settings.rate_limit_window_s
RATE_LIMIT_MAX = settings.rate_limit_max_requests


class RateLimitMiddleware(BaseHTTPMiddleware):
    """基于 Redis sorted set 的滑动窗口速率限制。

    仅拦截 /chat、/chat/stream；/health 和 /docs 等路径不受限。
    Redis 不可用时 fail-open（放行请求），记录 warning 日志。
    """

    async def dispatch(self, request: Request, call_next):
        # 仅限制 chat 端点
        if not request.url.path.startswith("/chat"):
            return await call_next(request)

        # 获取客户端 IP
        client_ip = request.client.host if request.client else "unknown"
        key = f"ratelimit:{client_ip}:chat"

        try:
            cache_store = getattr(request.app.state, "cache_store", None)
            if cache_store is None or not cache_store._redis:
                # 无 Redis → fail-open
                logger.warning("rate_limit_no_redis", client_ip=client_ip)
                return await call_next(request)

            now_ms = int(time.time() * 1000)
            window_start = now_ms - RATE_LIMIT_WINDOW * 1000

            redis = cache_store._redis

            # 滑动窗口：移除窗口外的旧记录 + 计数 + 添加当前请求
            async with redis.pipeline() as pipe:
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zcard(key)
                pipe.zadd(key, {str(now_ms): now_ms})
                pipe.expire(key, RATE_LIMIT_WINDOW + 1)
                results = await pipe.execute()

            current_count: int = results[1]  # zcard 在过期前的计数

            if current_count >= RATE_LIMIT_MAX:
                retry_after = RATE_LIMIT_WINDOW
                logger.warning(
                    "rate_limit_exceeded",
                    client_ip=client_ip,
                    count=current_count,
                    limit=RATE_LIMIT_MAX,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Too many requests. Try again in {retry_after} seconds."
                    },
                    headers={"Retry-After": str(retry_after)},
                )

        except Exception as e:
            # Redis 异常 → fail-open
            logger.warning("rate_limit_redis_unavailable", error=str(e))
            return await call_next(request)

        return await call_next(request)
