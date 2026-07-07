"""Compress 节点 — LLM 上下文压缩 (summary/analytical 意图)."""

import logging

from src.observability.langfuse_context import langfuse_context

from src.graph.state import RAGState
from src.rag.generation.compressor import compress
from src.observability.decorators import trace_rag_node

logger = logging.getLogger(__name__)


@trace_rag_node(name="compress_context")
async def compress_context(state: RAGState) -> dict:
    """上下文压缩：提取与 query 相关的句子。

    修复：压缩结果写入独立字段 compressed_context，
    不覆盖原始 documents，generate 节点优先使用压缩内容，
    source/metadata 信息保持完整。
    """

    intent = state.get("intent", "factoid")
    if intent not in ("summary", "analytical", "tabular"):
        return {}

    query = state.get("query", "")
    documents = state.get("documents") or []

    if not documents:
        return {}

    try:
        # Calculate original character count before compression
        original_chars = sum(len(d.get("content", "")) for d in documents)

        compressed = await compress(
            query,
            documents,
            fail_silently=True,
            model_name=state.get("model_name"),
        )
        logger.info(
            "[compress_context] 压缩完成: %d docs → %d chars",
            len(documents), len(compressed),
        )
        langfuse_context.update_current_observation(
            node="compress_context",
            original_chars=original_chars,
            compressed_chars=len(compressed),
            compression_triggered=True,
            doc_count=len(documents),
        )
        # 写入独立字段，不覆盖 documents
        return {"compressed_context": compressed}
    except Exception as exc:
        logger.warning("[compress_context] 压缩失败: %s → 跳过压缩", exc)
        langfuse_context.update_current_observation(
            node="compress_context",
            compression_triggered=False,
            error=str(exc),
        )
        return {"errors": [f"compress_context: {exc}"]}
