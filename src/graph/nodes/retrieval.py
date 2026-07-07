"""分层检索节点 — Tier1 (轻量) + Tier2 (增强).

Tier1: Dense 语义检索 + 关键词增强重排序（BM25 同款效果，零额外延迟）。
Tier2: HyDE + Hybrid (dense+sparse BM25) + RRF 融合。

连接管理：所有检索复用 dense_search 的持久 VectorStore 连接。
"""

import logging
import asyncio
from typing import Any

from src.observability.langfuse_context import langfuse_context

from src.graph.state import RAGState
from src.rag.retrieval.dense import dense_search, _get_vector_store
from src.rag.retrieval.hybrid import hybrid_search
from src.rag.retrieval.rrf import reciprocal_rank_fusion
from src.rag.retrieval.query_expansion import expand_query
from src.rag.embeddings.text_embedding_v4 import TextEmbeddingV4
from src.rag.retrieval.sparse import SparseRetriever
from src.rag.retrieval.collections import kb_collection_name, session_collection_name
from src.rag.retrieval.keyword_boost import extract_keywords, keyword_match_score
from src.rag.retrieval.query_processor import QueryProcessor
from src.config import settings
from src.observability.decorators import trace_rag_node

logger = logging.getLogger(__name__)

# 全局 embedding 实例（单例复用）
_embedder: TextEmbeddingV4 | None = None

# Per-collection sparse indexes — keyed by ChromaDB collection name
_sparse_indexes: dict[str, SparseRetriever] = {}
_sparse_indexes_ready: set[str] = set()
_sparse_index_lock = asyncio.Lock()
DEFAULT_COLLECTION = "rag_docs_dev"


def _keyword_boost_rerank(query: str, documents: list[dict]) -> list[dict]:
    """对 Tier1 检索结果做轻量关键词增强重排序.

    混合原始向量分数 (0.75) + 关键词匹配分数 (0.25)，
    提升精确关键词（产品名/实体名）匹配文档的排名。
    """
    keywords = extract_keywords(query)
    if not keywords or len(documents) <= 1:
        return documents

    boosted = []
    for doc in documents:
        content = doc.get("content", "")
        vector_score = doc.get("score", 0.0)

        kw_score = keyword_match_score(keywords=keywords, doc_content=content)
        # 混合: 75% 语义分数 + 25% 关键词分数 (降权以优先语义)
        blended = 0.75 * vector_score + 0.25 * kw_score
        boosted.append({**doc, "score": blended, "kw_match_score": round(kw_score, 4)})

    boosted.sort(key=lambda d: d.get("score", 0), reverse=True)
    return boosted


def _get_embedder() -> TextEmbeddingV4:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbeddingV4()
    return _embedder


async def _build_collection_sparse_index(col_name: str) -> SparseRetriever | None:
    """Build a BM25 sparse index for a single ChromaDB collection.

    Returns the SparseRetriever or None on failure.
    Uses parent_content when available for better context alignment.
    """
    global _sparse_indexes_ready

    if col_name in _sparse_indexes_ready and col_name in _sparse_indexes:
        return _sparse_indexes[col_name]

    try:
        vs = await _get_vector_store()
        col = await vs.get_or_create_collection(col_name)

        all_docs = await col.get(include=["documents", "metadatas"])
        if not (all_docs and all_docs.get("documents") and len(all_docs["documents"]) > 0):
            logger.info("[sparse_index] collection '%s' is empty, skipping", col_name)
            return None

        doc_list = []
        seen_parent_ids = set()

        for i in range(len(all_docs["documents"])):
            meta = all_docs["metadatas"][i] if all_docs.get("metadatas") else {}
            parent_id = meta.get("parent_chunk_id", "")
            parent_content = meta.get("parent_content", "")

            if parent_id and parent_content:
                # 每个 parent 只索引一次，避免重复
                if parent_id in seen_parent_ids:
                    continue
                seen_parent_ids.add(parent_id)
                doc_list.append({
                    "id": parent_id,
                    "content": parent_content,
                    "metadata": meta,
                })
            else:
                # 无 parent_content 时降级使用 child 文本
                doc_list.append({
                    "id": all_docs["ids"][i] if all_docs.get("ids") else str(i),
                    "content": all_docs["documents"][i],
                    "metadata": meta,
                })

        if doc_list:
            sp = SparseRetriever(collection_name=col_name)
            # Offload CPU-bound BM25 index construction to a thread executor
            # to avoid blocking the asyncio event loop on large collections
            await asyncio.get_event_loop().run_in_executor(
                None, sp.build_index, doc_list,
            )
            # 持久化到 JSON 文件
            await asyncio.get_event_loop().run_in_executor(
                None, sp.save_index, col_name,
            )
            _sparse_indexes[col_name] = sp
            _sparse_indexes_ready.add(col_name)
            logger.info(
                "[sparse_index] BM25 for '%s': parent_docs=%d (raw_chunks=%d)",
                col_name, len(doc_list), len(all_docs["documents"]),
            )
            return sp
        else:
            logger.warning("[sparse_index] collection '%s' has no valid docs", col_name)
            return None

    except Exception as exc:
        logger.warning("[sparse_index] build failed for '%s': %s", col_name, exc)
        return None


async def _ensure_sparse_indexes(
    kb_ids: list[str] | None = None,
    session_id: str | None = None,
) -> list[tuple[str, SparseRetriever]]:
    """Ensure sparse indexes exist for the requested collections.

    Returns list of (collection_name, SparseRetriever) ready for searching.
    """
    col_names: list[str] = []

    if kb_ids:
        col_names.extend(kb_collection_name(str(kid)) for kid in kb_ids)
    else:
        col_names.append(DEFAULT_COLLECTION)

    if session_id:
        col_names.append(session_collection_name(str(session_id)))

    results: list[tuple[str, SparseRetriever]] = []
    for col_name in col_names:
        async with _sparse_index_lock:
            sp = await _build_collection_sparse_index(col_name)
        if sp is not None:
            results.append((col_name, sp))

    return results


async def warmup_sparse_index() -> None:
    """应用启动时预热默认 BM25 索引，避免首次 Tier2 查询冷启动延迟。"""
    try:
        async with _sparse_index_lock:
            await _build_collection_sparse_index(DEFAULT_COLLECTION)
        logger.info("[sparse_index] 启动预热完成")
    except Exception as exc:
        logger.warning("[sparse_index] 启动预热失败: %s", exc)


async def invalidate_sparse_index(col_name: str | None = None) -> None:
    """Mark BM25 indexes as stale after document/KB lifecycle changes.

    Clears in-memory cache AND deletes persisted JSON files.

    Args:
        col_name: Specific collection to invalidate. If None, invalidates all.
    """
    global _sparse_indexes_ready
    async with _sparse_index_lock:
        if col_name:
            _sparse_indexes_ready.discard(col_name)
            _sparse_indexes.pop(col_name, None)
            # 删除持久化的 BM25 JSON 文件
            _delete_bm25_file(col_name)
            logger.info("[sparse_index] invalidated collection: %s", col_name)
        else:
            # 收集所有已知的 collection 名称以便删除文件
            all_collections = list(_sparse_indexes.keys()) + list(_sparse_indexes_ready)
            _sparse_indexes_ready.clear()
            _sparse_indexes.clear()
            for col in set(all_collections):
                _delete_bm25_file(col)
            logger.info("[sparse_index] all indexes invalidated")


def _delete_bm25_file(col_name: str) -> None:
    """删除指定 collection 的 BM25 JSON 持久化文件."""
    import os
    from src.rag.retrieval.sparse import _index_path, _tmp_path
    for path in (_index_path(col_name), _tmp_path(col_name)):
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.info("[sparse_index] bm25 file deleted: %s", path)
            except OSError as exc:
                logger.warning("[sparse_index] failed to delete bm25 file: %s, error=%s", path, exc)


async def _embed_query(query: str) -> list[float]:
    """将查询文本转为 embedding 向量."""
    embedder = _get_embedder()
    result = await embedder.embed([query])
    return result[0] if result else []


@trace_rag_node(name="tier1_retrieve")
async def tier1_retrieve(state: RAGState) -> dict:
    """Tier1 检索：Dense 语义检索 + 关键词增强重排序。

    1. 复用持久 VectorStore 连接（不再每次 connect/close）
    2. 先取 retrieval_top_k × 2 的候选（为关键词重排提供更多选择）
    3. 用关键词匹配对候选做轻量重排序
    4. 截断到 retrieval_top_k 返回
    5. 所有意图统一配置，由后段 rerank 按意图裁剪。
    6. 支持 key:value 查询过滤语法（collection/type/tag/source）。
    """

    query = state.get("query", "")
    kb_ids = state.get("kb_ids") or []
    session_id = state.get("session_id")

    # ── 查询过滤解析 ──
    parsed = QueryProcessor.parse(query)
    where = QueryProcessor.to_chromadb_where(parsed.filters)
    filter_collection = QueryProcessor.get_collection_name(parsed.filters)
    semantic_query = parsed.clean_query or query  # 使用清洗后的查询

    # Apply suggested_k from intent router with sensible min/max clamps
    suggested_k = state.get("suggested_k", 0)
    if suggested_k and suggested_k > 0:
        top_k = max(5, min(suggested_k, settings.retrieval_top_k * 2))
    else:
        top_k = settings.retrieval_top_k

    try:
        embedding = await _embed_query(semantic_query)
        if not embedding:
            langfuse_context.update_current_observation(
                node="tier1_retrieve",
                retrieval_type="dense",
                top_k=top_k,
                recall_count=0,
                error="embedding_failed",
            )
            return {"documents": [], "retrieval_scores": [], "retrieval_tier": 1}

        # 使用持久连接进行检索
        vs = await _get_vector_store()

        # 如果有 collection filter，限制检索范围
        kb_ids_list = list(kb_ids) if kb_ids else []
        if filter_collection and not kb_ids_list:
            # 若用户指定了 collection filter 且未传入 kb_ids，则将 collection 名添加到检索
            kb_ids_list = [filter_collection]

        # 先取 2x 候选，给关键词重排提供空间
        candidates = await dense_search(
            embedding, top_k=max(top_k * 2, 10),
            vector_store=vs, kb_ids=kb_ids_list, session_id=session_id,
            where=where,
        )

        # 关键词增强重排序（使用清洗后的语义查询）
        reranked = _keyword_boost_rerank(semantic_query, candidates)

        # 截断到 top_k
        results = reranked[:top_k]

        scores = [r.get("score", 0.0) for r in results]
        logger.info(
            "[tier1_retrieve] dense(%d) + keyword boost → top %d, query='%s' (filters=%s)",
            len(candidates), len(results), semantic_query[:50],
            {k: v for k, v in parsed.filters.items()},
        )
        langfuse_context.update_current_observation(
            node="tier1_retrieve",
            retrieval_type="dense_keyword_boost",
            top_k=top_k,
            candidates_count=len(candidates),
            recall_count=len(results),
            query_filters={k: v for k, v in parsed.filters.items()},
        )
        return {
            "documents": results,
            "retrieval_scores": scores,
            "retrieval_tier": 1,
            "query_filters": parsed.filters,
            "clean_query": semantic_query,
        }
    except Exception as exc:
        logger.warning("[tier1_retrieve] 检索失败: %s", exc)
        langfuse_context.update_current_observation(
            node="tier1_retrieve",
            retrieval_type="dense",
            top_k=top_k,
            recall_count=0,
            error=str(exc),
        )
        return {
            "documents": [],
            "retrieval_scores": [],
            "retrieval_tier": 1,
            "errors": [f"tier1_retrieve: {exc}"],
        }


@trace_rag_node(name="tier2_retrieve")
async def tier2_retrieve(state: RAGState) -> dict:
    """Tier2 增强检索：+HyDE → Hybrid → RRF 融合。

    在 Tier1 Gate fail 时触发。
    支持 key:value 查询过滤语法（collection/type/tag/source）。
    """

    query = state.get("query", "")
    existing_docs = state.get("documents") or []
    kb_ids = state.get("kb_ids") or []
    session_id = state.get("session_id")

    # ── 查询过滤解析 ──
    parsed = QueryProcessor.parse(query)
    where = QueryProcessor.to_chromadb_where(parsed.filters)
    filter_collection = QueryProcessor.get_collection_name(parsed.filters)
    semantic_query = parsed.clean_query or query

    # Apply suggested_k from intent router with sensible min/max clamps
    suggested_k = state.get("suggested_k", 0)
    if suggested_k and suggested_k > 0:
        search_top_k = max(5, min(suggested_k, settings.retrieval_top_k * 2))
    else:
        search_top_k = 20

    try:
        # HyDE 查询拓展（失败不阻塞，使用语义查询）
        expanded_query = semantic_query
        try:
            hyde_result = await expand_query(semantic_query, model_name=state.get("model_name"))
            if hyde_result:
                expanded_query = hyde_result
        except Exception:
            pass

        # 嵌入拓展后的查询
        embedding = await _embed_query(expanded_query)
        if not embedding:
            logger.warning("[tier2_retrieve] embedding 失败，使用已有结果")
            langfuse_context.update_current_observation(
                node="tier2_retrieve",
                hyde_applied=expanded_query != semantic_query,
                embedding_failed=True,
                final_doc_count=len(existing_docs),
            )
            return {"documents": existing_docs, "retrieval_tier": 2}

        # 复用持久 VectorStore 连接
        vs = await _get_vector_store()

        kb_ids_list = list(kb_ids) if kb_ids else []
        if filter_collection and not kb_ids_list:
            kb_ids_list = [filter_collection]

        # Dense search (multi-KB, already collection-aware, with filters)
        dense_results = await dense_search(
            embedding, top_k=search_top_k, vector_store=vs,
            kb_ids=kb_ids_list if kb_ids_list else None,
            session_id=session_id,
            where=where,
        )

        # Sparse search (collection-aware — per KB/session collection)
        sparse_index_pairs = await _ensure_sparse_indexes(
            kb_ids=kb_ids_list if kb_ids_list else None,
            session_id=session_id,
        )
        sparse_results: list[dict] = []
        seen_sparse_ids = set()
        for col_name, sp in sparse_index_pairs:
            col_results = await asyncio.get_event_loop().run_in_executor(
                None, sp.search, semantic_query, search_top_k,
            )
            for doc in col_results:
                key = doc.get("id", "")
                if key and key not in seen_sparse_ids:
                    seen_sparse_ids.add(key)
                    sparse_results.append(doc)

        # RRF 融合 (dense + collection-aware sparse)
        result_lists = [dense_results, sparse_results]
        if existing_docs:
            result_lists.append(existing_docs)

        fused = reciprocal_rank_fusion(result_lists, k=60)

        scores = [r.get("score", 0.0) for r in fused]
        langfuse_context.update_current_observation(
            node="tier2_retrieve",
            hyde_applied=expanded_query != semantic_query,
            dense_count=len(dense_results),
            bm25_count=len(sparse_results),
            existing_count=len(existing_docs),
            final_doc_count=len(fused),
            rrf_k=60,
            query_filters={k: v for k, v in parsed.filters.items()},
        )
        return {
            "documents": fused,
            "retrieval_scores": scores,
            "retrieval_tier": 2,
            "query_filters": parsed.filters,
            "clean_query": semantic_query,
        }
    except Exception as exc:
        logger.warning("[tier2_retrieve] 检索失败: %s → 使用已有结果", exc)
        langfuse_context.update_current_observation(
            node="tier2_retrieve",
            hyde_applied=False,
            final_doc_count=len(existing_docs),
            error=str(exc),
        )
        return {
            "documents": existing_docs,
            "retrieval_scores": [],
            "retrieval_tier": 2,
            "errors": [f"tier2_retrieve: {exc}"],
        }

