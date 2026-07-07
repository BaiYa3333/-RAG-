"""Quality Gate 节点 — 规则引擎判定检索质量."""

import logging

from src.observability.langfuse_context import langfuse_context

from src.graph.state import RAGState
from src.config import settings
from src.observability.decorators import trace_rag_node

logger = logging.getLogger(__name__)

# Tier1 向量相似度阈值（cosine similarity，值域 0~1），由 RAG_TIER1_SCORE_THRESHOLD 环境变量控制
TIER1_SCORE_THRESHOLD = settings.tier1_score_threshold

# Tier2 RRF 分数阈值（值域约 0.01~0.05，top1 约 0.016）
# 只要有结果且最高分 > 此值即视为有效
TIER2_SCORE_THRESHOLD = 0.008

# 最少文档数
MIN_DOC_COUNT = 1


@trace_rag_node(name="quality_gate")
async def check_quality(state: RAGState) -> dict:
    """检查检索质量，按 tier 使用不同阈值判断。

    Tier1：avg cosine similarity > 0.45 AND doc_count >= 1 → pass
    Tier2：max rrf_score > 0.008 AND doc_count >= 1 → pass
           （Tier2 已是最后手段，有结果就放行）
    """

    documents = state.get("documents") or []
    tier = state.get("retrieval_tier", 1)
    doc_count = len(documents)

    if doc_count == 0:
        action = "escalate" if tier == 1 else "fallback_generate"
        reason = "no_results"
        avg_score = 0.0
        max_score = 0.0
    else:
        if tier == 1:
            # Tier1：使用 retrieval_scores（向量相似度）
            scores = state.get("retrieval_scores") or [
                d.get("score", 0.0) for d in documents
            ]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0
            if avg_score >= TIER1_SCORE_THRESHOLD:
                action = "pass"
                reason = "quality_sufficient"
            else:
                action = "escalate"
                reason = "low_similarity"
        else:
            # Tier2：使用 rrf_score（量纲不同，取 max 判断阈值，avg 真实记录）
            rrf_scores = [
                d.get("rrf_score", d.get("score", 0.0)) for d in documents
            ]
            max_score = max(rrf_scores) if rrf_scores else 0.0
            avg_score = sum(rrf_scores) / len(rrf_scores) if rrf_scores else 0.0
            if max_score >= TIER2_SCORE_THRESHOLD and doc_count >= MIN_DOC_COUNT:
                action = "pass"
                reason = "tier2_quality_sufficient"
            else:
                action = "fallback_generate"
                reason = "tier2_low_score"

    gate_entry = {
        "tier": tier,
        "avg_score": round(avg_score, 4),
        "max_score": round(max_score, 4),
        "doc_count": doc_count,
        "reason": reason,
        "action": action,
    }

    langfuse_context.update_current_observation(
        node="quality_gate",
        tier=tier,
        action=action,
        avg_score=round(avg_score, 4),
        threshold=TIER1_SCORE_THRESHOLD if tier == 1 else TIER2_SCORE_THRESHOLD,
        doc_count=doc_count,
        reason=reason,
    )

    logger.info(
        "[quality_gate] tier=%d score=%.4f count=%d → %s",
        tier, avg_score, doc_count, action,
    )

    return {"gate_log": [gate_entry]}
