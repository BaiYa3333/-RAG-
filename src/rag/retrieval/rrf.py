"""RRF 融合 — Reciprocal Rank Fusion, k=60.

统一去重键：优先使用 metadata.parent_chunk_id 确保 dense (child chunk_id) 和
sparse (parent_chunk_id) 的同一父块被正确去重，避免父块因子块多数而分数虚高。
"""

from src.config import settings


def _get_rrf_key(doc: dict) -> str:
    """获取 RRF 去重键 — 统一用 parent_chunk_id，回退到 id。"""
    meta = doc.get("metadata", {})
    if isinstance(meta, dict) and meta.get("parent_chunk_id"):
        return meta["parent_chunk_id"]
    return doc.get("id", "")


def reciprocal_rank_fusion(result_lists: list[list[dict]],
                           k: int | None = None, top_k: int | None = None) -> list[dict]:
    if k is None:
        k = settings.rrf_k
    if top_k is None:
        top_k = settings.rrf_top_k

    scores: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = _get_rrf_key(doc)
            if doc_id not in scores:
                scores[doc_id] = {**doc, "rrf_score": 0.0}
            scores[doc_id]["rrf_score"] += 1.0 / (k + rank + 1)

    fused = sorted(scores.values(), key=lambda d: d["rrf_score"], reverse=True)
    result = fused[:top_k]

    # Record RRF metadata
    try:
        from src.observability.langfuse_context import langfuse_context
        langfuse_context.update_current_observation(
            retrieval_type="rrf_fusion",
            input_lists=len(result_lists),
            input_total_docs=sum(len(r) for r in result_lists),
            fused_count=len(fused),
            final_count=len(result),
            rrf_k=k,
        )
    except Exception:
        pass

    return result
