"""text-embedding-v4 — DashScope embedding API (OpenAI-compatible) + Redis 缓存."""

import asyncio
import hashlib
import json

import httpx
from openai import AsyncOpenAI

from src.config import settings
from src.utils.logger import logger

# 缓存 TTL（1 小时）
EMBED_CACHE_TTL = 3600
CACHE_MIN_CHARS = 4     # 低于此长度的文本不缓存
CACHE_MAX_CHARS = 2000  # 扩大到覆盖 parent_chunk (1024字)，提升文档 embedding 缓存命中率


_shared_cache_store = None
_shared_cache_lock = asyncio.Lock()


async def close_embedding_cache() -> None:
    """Close the internally owned embedding cache, if it was created."""
    global _shared_cache_store
    if _shared_cache_store is not None:
        try:
            await _shared_cache_store.close()
        finally:
            _shared_cache_store = None


async def _get_shared_cache_store():
    """Return a singleton Redis CacheStore for embedding cache use."""
    global _shared_cache_store
    if _shared_cache_store is not None and _shared_cache_store._redis is not None:
        return _shared_cache_store

    async with _shared_cache_lock:
        if _shared_cache_store is not None and _shared_cache_store._redis is not None:
            return _shared_cache_store
        try:
            from src.stores.cache_store import CacheStore

            store = CacheStore()
            await store.connect()
            _shared_cache_store = store
            return _shared_cache_store
        except Exception:
            _shared_cache_store = None
            return None


class TextEmbeddingV4:
    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            http_client=httpx.AsyncClient(timeout=30.0),
        )
        self._model = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._cache_store = None  # 使用模块级单例缓存连接

    def _get_cache_store(self):
        """延迟获取 Redis cache store（避免循环导入，持久连接）."""
        if self._cache_store is not None and self._cache_store._redis is not None:
            return self._cache_store
        try:
            from src.stores.cache_store import CacheStore
            store = CacheStore()
            self._cache_store = store
        except Exception:
            return None
        return self._cache_store

    async def _ensure_redis_connected(self):
        """获取应用生命周期内复用的 Redis cache store。"""
        store = await _get_shared_cache_store()
        self._cache_store = store
        return store

    def _cache_key(self, text: str) -> str:
        """为查询文本生成缓存键."""
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"emb:{self._model}:{h}"

    async def _try_cache(self, texts: list[str]) -> tuple[dict[str, list[float]], list[str]]:
        """尝试从 Redis 缓存获取 embeddings。返回 (cached_map, missed_texts).

        支持批量请求 — 逐条检查缓存，已缓存的文本跳过 API 调用。
        使用 zip 将每个文本与其缓存键显式绑定，消除隐式索引对齐。
        """
        try:
            store = await self._ensure_redis_connected()
            if store is None:
                return {}, list(texts)

            cached_map: dict[str, list[float]] = {}
            missed: list[str] = []

            # Build (text, cache_key) pairs for cacheable texts — zip binds
            # each text to its key explicitly, avoiding fragile index alignment
            cacheable: list[tuple[str, str]] = []
            for text in texts:
                if len(text) < CACHE_MIN_CHARS or len(text) > CACHE_MAX_CHARS:
                    missed.append(text)
                    continue
                cacheable.append((text, self._cache_key(text)))

            if not cacheable:
                return {}, list(texts)

            # 批量获取缓存
            pipe = store._redis.pipeline()
            for _, key in cacheable:
                pipe.get(key)
            results = await pipe.execute()

            # Map results back to texts using zip for explicit pairing
            for (text, _key), cached in zip(cacheable, results):
                if cached:
                    cached_map[text] = json.loads(cached)
                    logger.debug("[embed_cache] hit: '%s...'", text[:30])
                else:
                    missed.append(text)

            return cached_map, missed
        except Exception:
            return {}, list(texts)

    async def _set_cache(self, text: str, embedding: list[float]) -> None:
        """将单个 embedding 写入 Redis 缓存."""
        if len(text) < CACHE_MIN_CHARS or len(text) > CACHE_MAX_CHARS:
            return

        try:
            store = await self._ensure_redis_connected()
            if store is None:
                return

            key = self._cache_key(text)
            await store._redis.set(key, json.dumps(embedding), ex=EMBED_CACHE_TTL)
        except Exception:
            pass

    async def _set_cache_batch(self, text_emb_pairs: list[tuple[str, list[float]]]) -> None:
        """批量写入 embedding 缓存（pipeline 提升写入效率）."""
        valid = [(t, e) for t, e in text_emb_pairs
                 if CACHE_MIN_CHARS <= len(t) <= CACHE_MAX_CHARS]
        if not valid:
            return

        try:
            store = await self._ensure_redis_connected()
            if store is None:
                return

            pipe = store._redis.pipeline()
            for text, embedding in valid:
                key = self._cache_key(text)
                pipe.set(key, json.dumps(embedding), ex=EMBED_CACHE_TTL)
            await pipe.execute()
            logger.debug("[embed_cache] batch set: %d entries", len(valid))
        except Exception:
            pass

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # 批量缓存检查（支持多文本 + 长文本缓存）
        cached_map, missed = await self._try_cache(texts)
        if not missed:
            return [cached_map[t] for t in texts]

        # 对未命中的文本调用 API
        if len(missed) <= self._batch_size:
            embeddings = await self._embed_batch(missed)
        else:
            batches = [missed[i:i + self._batch_size] for i in range(0, len(missed), self._batch_size)]
            results = await asyncio.gather(*(self._embed_batch(b) for b in batches))
            embeddings = [v for batch in results for v in batch]

        # 批量写入缓存
        await self._set_cache_batch(list(zip(missed, embeddings)))

        # 重组结果（保持原始顺序）
        result_map = {**cached_map, **dict(zip(missed, embeddings))}
        return [result_map[t] for t in texts]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(3):
            try:
                resp = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                )
                return [d.embedding for d in resp.data]
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning("embed_retry", attempt=attempt + 1, wait_s=wait, error=str(e))
                await asyncio.sleep(wait)

        raise RuntimeError("unreachable")
