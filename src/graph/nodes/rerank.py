"""Rerank 节点 — Cross-encoder 重排序 (DashScope gte-rerank)."""

import logging

from src.observability.langfuse_context import langfuse_context

from src.graph.state import RAGState
from src.config import settings
from src.observability.decorators import trace_rag_node

logger = logging.getLogger(__name__)


# 意图 → rerank top_k 映射: summary/analytical 需要更多文档
RERANK_K_MAP = {
    "factoid": None,       # None = 使用 settings.rerank_top_k
    "comparison": None,
    "summary": None,       # 改用 settings.rerank_top_k * 2
    "analytical": None,
}


@trace_rag_node(name="rerank_docs")
async def rerank_docs(state: RAGState) -> dict:
    """Cross-encoder 重排序。按意图动态取 top_k：summary/analytical 取 2x，其余取 1x."""

    documents = state.get("documents") or []

    if not documents:
        return {"rerank_applied": False, "rerank_method": "none"}

    intent = state.get("intent", "factoid")
    if intent in ("summary", "analytical", "tabular"):
        top_k = settings.rerank_top_k * 2
    else:
        top_k = settings.rerank_top_k

    try:
        from src.rag.retrieval.reranker import rerank

        reranked = await rerank(
            query=state.get("query", ""),
            documents=documents,
            top_k=top_k,
            model_name=state.get("model_name"),
        )
        rerank_method = "none"
        if reranked:
            rerank_method = reranked[0].get("_rerank_method", "api")
        rerank_applied = rerank_method in ("api", "llm")
        cleaned = [
            {k: v for k, v in doc.items() if k != "_rerank_method"}
            for doc in reranked
        ]
        langfuse_context.update_current_observation(
            node="rerank_docs",
            rerank_method=rerank_method,
            input_count=len(documents),
            output_count=len(cleaned),
            model_name=settings.rerank_model,
        )
        return {
            "documents": cleaned,
            "rerank_applied": rerank_applied,
            "rerank_method": rerank_method,
        }
    except Exception as exc:
        logger.warning("[rerank_docs] 重排序失败: %s → 跳过 rerank", exc)
        langfuse_context.update_current_observation(
            node="rerank_docs",
            rerank_method="none",
            input_count=len(documents),
            output_count=0,
            error=str(exc),
        )
        return {
            "errors": [f"rerank_docs: {exc}"],
            "rerank_applied": False,
            "rerank_method": "none",
        }
