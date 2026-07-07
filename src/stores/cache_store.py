"""Redis 缓存存储 — redis.asyncio 封装."""

import redis.asyncio as aioredis

from src.config import settings
from src.utils.logger import logger


class CacheStore:
    def __init__(self):
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._redis = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
        )
        logger.info("redis_connected", host=settings.redis_host, port=settings.redis_port)

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
        logger.info("redis_closed")

    async def health(self) -> bool:
        return await self._redis.ping()

    async def get(self, key: str) -> str | None:
        value = await self._redis.get(key)
        return value.decode("utf-8") if value else None

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        if ttl:
            await self._redis.setex(key, ttl, value)
        else:
            await self._redis.set(key, value)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def exists(self, key: str) -> bool:
        return await self._redis.exists(key) > 0
