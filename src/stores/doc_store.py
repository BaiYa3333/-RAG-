"""PostgreSQL 文档存储 — asyncpg 连接池封装."""

import asyncpg

from src.config import settings
from src.utils.logger import logger


class DocStore:
    def __init__(self):
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            dsn=settings.postgres_dsn,
            min_size=settings.postgres_min_pool,
            max_size=settings.postgres_max_pool,
        )
        logger.info("postgres_connected", dsn=settings.postgres_dsn)

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "PostgreSQL is not connected. Check RAG_POSTGRES_DSN and startup logs."
            )
        return self._pool

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
        logger.info("postgres_closed")

    async def health(self) -> int:
        async with self._require_pool().acquire() as conn:
            return await conn.fetchval("SELECT 1")

    async def fetchval(self, query: str, *args):
        async with self._require_pool().acquire() as conn:
            return await conn.fetchval(query, *args)

    async def fetchrow(self, query: str, *args) -> dict | None:
        async with self._require_pool().acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def fetch(self, query: str, *args) -> list[dict]:
        async with self._require_pool().acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(r) for r in rows]

    async def execute(self, query: str, *args) -> str:
        async with self._require_pool().acquire() as conn:
            return await conn.execute(query, *args)

    async def acquire(self) -> asyncpg.Connection:
        return await self._require_pool().acquire()

    # ── documents 表专用 ──────────────────────────────────────

    async def insert_document(self, title: str, source: str, doc_type: str,
                              file_hash: str, chunk_count: int = 0,
                              metadata: dict | None = None,
                              kb_id: str | None = None) -> str:
        import json
        query = """
            INSERT INTO documents (title, source, doc_type, file_hash, chunk_count, metadata, kb_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7::uuid)
            RETURNING id
        """
        return await self.fetchval(
            query, title, source, doc_type, file_hash, chunk_count,
            json.dumps(metadata or {}), kb_id,
        )

    async def get_document(self, doc_id: str) -> dict | None:
        query = "SELECT * FROM documents WHERE id = $1"
        return await self.fetchrow(query, doc_id)
