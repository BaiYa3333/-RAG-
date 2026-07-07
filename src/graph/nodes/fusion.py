"""RRF Fusion 节点 — 多路检索结果融合（当前未挂载到主 workflow，由 tier2_retrieve 内部调用）."""

import logging
from src.graph.state import RAGState
from src.rag.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


async def fuse_results(state: RAGState) -> dict:
    """RRF 融合：合并 dense + sparse 多路检索结果。

    注意：当前主 workflow 中 tier2_retrieve 内部已完成 RRF 融合，
    本节点保留用于未来独立 hybrid 路径扩展。
    """
    documents = state.get("documents") or []
    sparse_docs = state.get("sparse_documents") or []

    if not documents and not sparse_docs:
        return {}

    result_lists = [lst for lst in [documents, sparse_docs] if lst]
    if len(result_lists) == 1:
        # 单路无需融合
        return {"documents": documents}

    try:
        fused = reciprocal_rank_fusion(result_lists, k=60)
        scores = [r.get("rrf_score", r.get("score", 0.0)) for r in fused]
        return {
            "documents": fused,
            "retrieval_scores": scores,
        }
    except Exception as exc:
        logger.warning("[fuse_results] RRF 失败: %s → 保持原结果", exc)
        return {"errors": [f"fuse_results: {exc}"]}
