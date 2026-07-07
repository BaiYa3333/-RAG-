"""稠密检索 — ChromaDB 向量相似度搜索（连接池复用）."""

import asyncio
import logging
import time

from src.observability.langfuse_context import langfuse_context

from src.config import settings
from src.rag.retrieval.collections import kb_collection_name, session_collection_name
from src.stores.vector_store import VectorStore
from src.observability.decorators import trace_rag_node

logger = logging.getLogger(__name__)

# 模块级持久连接 — 避免每次检索都重新建立 ChromaDB HTTP 连接
_vs: VectorStore | None = None
_vs_lock = asyncio.Lock()
_col_name = "rag_docs_dev"


async def _get_vector_store() -> VectorStore:
    """获取或创建持久 VectorStore 连接（线程安全懒加载）."""
    global _vs
    if _vs is not None and _vs._client is not None:
        return _vs

    async with _vs_lock:
        if _vs is not None and _vs._client is not None:
            return _vs
        _vs = VectorStore()
        await _vs.connect()
        logger.info("[dense] VectorStore 持久连接已建立")
        return _vs


async def _search_collection(
    vs: VectorStore,
    col_name: str,
    query_embedding: list[float],
    top_k: int,
    where: dict | None = None,
    source_label: str = "dense",
) -> list[dict]:
    """Search a single collection and return formatted docs. Gracefully handles missing collections."""
    try:
        col = await vs.get_or_create_collection(col_name)
        result = await vs.search(col, [query_embedding], top_k=top_k, where=where)

        docs = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            parent_content = meta.get("parent_content", "")
            child_content = documents[i] if i < len(documents) else ""
            content = parent_content if parent_content else child_content
            docs.append({
                "id": ids[i],
                "content": content,
                "metadata": meta,
                "score": 1 - distances[i] if i < len(distances) else 0,
                "source": source_label,
            })

        return docs
    except Exception as e:
        logger.warning("[dense] search collection '%s' failed: %s", col_name, e)
        return []


@trace_rag_node(name="dense_search")
async def dense_search(query_embedding: list[float], top_k: int | None = None,
                       where: dict | None = None,
                       vector_store: VectorStore | None = None,
                       kb_id: str | None = None,
                       kb_ids: list[str] | None = None,
                       session_id: str | None = None) -> list[dict]:
    k = top_k or settings.retrieval_top_k
    start_time = time.monotonic()

    # 优先使用传入的 vector_store，否则复用持久连接
    vs = vector_store if vector_store is not None else await _get_vector_store()

    collection_names: list[tuple[str, str]] = []
    selected_kb_ids = list(kb_ids or ([] if kb_id is None else [kb_id]))
    if selected_kb_ids:
        collection_names.extend((kb_collection_name(str(kid)), "dense") for kid in selected_kb_ids)
    else:
        collection_names.append((_col_name, "dense"))

    if session_id:
        collection_names.append((session_collection_name(str(session_id)), "session"))

    result_lists = await asyncio.gather(*(
        _search_collection(vs, col_name, query_embedding, k, where=where, source_label=source_label)
        for col_name, source_label in collection_names
    ))

    docs: list[dict] = []
    seen: set[str] = set()
    for result in result_lists:
        for doc in result:
            key = _dedupe_key(doc)
            if key in seen:
                continue
            seen.add(key)
            docs.append(doc)

    docs.sort(key=lambda d: d.get("score", 0), reverse=True)
    result = docs[:k]
    latency_ms = (time.monotonic() - start_time) * 1000
    langfuse_context.update_current_observation(
        retrieval_type="dense",
        top_k=k,
        recall_count=len(result),
        latency_ms=round(latency_ms, 2),
    )
    return result


def _dedupe_key(doc: dict) -> str:
    meta = doc.get("metadata") or {}
    for key in ("doc_id", "chunk_id", "parent_chunk_id"):
        value = meta.get(key)
        if value:
            return f"{key}:{value}"
    return str(doc.get("id", ""))


async def _cleanup_session_collection(session_id: str) -> bool:
    """Delete the session-scoped ChromaDB collection if it exists.

    Best-effort: failures are logged but not raised.
    Returns True if the collection was deleted, False otherwise.
    """
    col_name = session_collection_name(str(session_id))
    try:
        vs = await _get_vector_store()
        if vs._client is None:
            logger.warning("[dense] cleanup skipped: VectorStore client not connected")
            return False

        # ChromaDB AsyncHttpClient.delete_collection is the direct approach
        await vs._client.delete_collection(col_name)
        logger.info("[dense] session collection cleaned up: %s", col_name)
        return True
    except Exception as e:
        # Collection may not exist — that's fine
        logger.warning("[dense] cleanup session collection '%s' (best-effort): %s", col_name, e)
        return False


async def close_vector_store():
    """关闭持久 VectorStore 连接（应用关闭时调用）."""
    global _vs
    if _vs is not None:
        await _vs.close()
        _vs = None
        logger.info("[dense] VectorStore 持久连接已关闭")
