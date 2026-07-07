"""混合检索编排 — dense + sparse 并行，支持多 KB 和会话集合."""

import asyncio

from src.rag.retrieval.dense import dense_search
from src.rag.retrieval.metadata_filter import build_metadata_filter
from src.rag.retrieval.sparse import SparseRetriever


async def hybrid_search(query: str, query_embedding: list[float],
                        top_k: int = 20, sparse_retriever: SparseRetriever | None = None,
                        metadata_filters: dict | None = None,
                        kb_id: str | None = None,
                        kb_ids: list[str] | None = None,
                        session_id: str | None = None) -> dict:
    where = build_metadata_filter(**(metadata_filters or {}))

    # Support both legacy kb_id and new kb_ids for multi-KB search
    selected_kb_ids = list(kb_ids or ([] if kb_id is None else [kb_id]))

    dense_future = dense_search(
        query_embedding, top_k=top_k, where=where,
        kb_ids=selected_kb_ids if selected_kb_ids else None,
        session_id=session_id,
    )

    if sparse_retriever:
        loop = asyncio.get_event_loop()
        sparse_future = loop.run_in_executor(None, sparse_retriever.search, query, top_k)
        dense_results, sparse_results = await asyncio.gather(dense_future, sparse_future)
    else:
        dense_results = await dense_future
        sparse_results = []

    return {
        "dense": dense_results,
        "sparse": sparse_results,
    }
