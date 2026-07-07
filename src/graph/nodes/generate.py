"""Generate 节点 — LLM 流式/非流式生成."""

import logging

from src.observability.langfuse_context import langfuse_context

from src.graph.state import RAGState
from src.rag.generation.generator import generate, generate_stream
from src.observability.decorators import trace_rag_node

logger = logging.getLogger(__name__)

_CONTEXT_LIMITS = {
    "factoid": 8,
    "comparison": 8,
    "summary": 16,
    "analytical": 16,
    "tabular": 10,
}


def _doc_source(doc: dict) -> str:
    metadata = doc.get("metadata", {}) or {}
    return metadata.get("source", doc.get("source", ""))


def _doc_title(doc: dict) -> str:
    metadata = doc.get("metadata", {}) or {}
    return metadata.get("title", "")


@trace_rag_node(name="generate_answer")
async def generate_answer(state: RAGState) -> dict:
    """LLM 生成答案（流式 + fallback).

    优先使用 compressed_context（summary/analytical 意图），
    附加 source 信息保留引用，并记录实际送入生成器的上下文。
    """

    query = state.get("query", "")
    documents = state.get("documents") or []
    compressed_context = state.get("compressed_context", "")

    if not documents and not compressed_context:
        return {
            "answer": "未找到相关文档。请尝试重新表述您的问题，或上传更多相关文档。",
            "used_contexts": [],
            "used_context_count": 0,
            "used_sources": [],
        }

    intent = state.get("intent", "factoid")
    max_docs = _CONTEXT_LIMITS.get(intent, 8)

    # 按相似度/重排序分数降序排列，确保 used_contexts 与最终 prompt 一致。
    sorted_docs = sorted(
        documents,
        key=lambda d: d.get("rerank_score", d.get("rrf_score", d.get("score", 0))),
        reverse=True,
    )
    selected_docs = sorted_docs[:max_docs]
    used_sources = [s for d in selected_docs if (s := _doc_source(d))]

    # 优先使用压缩上下文（summary/analytical 意图）
    if compressed_context and intent in ("summary", "analytical"):
        context = compressed_context
        used_contexts = [compressed_context]
        # 附加原始文档的 source 信息（保留引用）
        sources_info = "\n".join(f"- {source}" for source in used_sources)
        if sources_info:
            context += f"\n\n[参考来源]\n{sources_info}"
    else:
        context_parts = []
        used_contexts = []
        for i, d in enumerate(selected_docs):
            content = d.get("content", "")
            if not content:
                continue
            source = _doc_source(d)
            title = _doc_title(d)
            header = f"[文档{i+1}]"
            if title:
                header += f" 来源: {title}"
            elif source:
                header += f" 来源: {source}"
            context_parts.append(f"{header}\n{content}")
            used_contexts.append(content)

        context = "\n\n---\n\n".join(context_parts)

    base_result = {
        "used_contexts": used_contexts,
        "used_context_count": len(used_contexts),
        "used_sources": used_sources,
    }

    # In stream_mode, prepare context for external generate_stream() call
    if state.get("stream_mode"):
        langfuse_context.update_current_observation(
            node="generate_answer",
            generation_mode="stream",
            context_size=len(context),
            doc_count=0,
        )
        return {
            "answer": "",  # filled by streaming endpoint
            "streaming_context": context,     # 不能用 _ 前缀 — LangGraph 会过滤
            "streaming_intent": intent,
            **base_result,
        }

    # 非流式生成
    try:
        answer = await generate(
            query,
            context,
            model_name=state.get("model_name"),
            language=state.get("language"),
            format_mode="auto",
            intent=intent,
        )
        # 解析引用
        from src.rag.generation.generator import parse_citations
        citations = parse_citations(answer, selected_docs)
        langfuse_context.update_current_observation(
            node="generate_answer",
            generation_mode="non_stream",
            context_size=len(context),
            doc_count=len(selected_docs),
            answer_len=len(answer),
        )
        return {"answer": answer, "citations": citations, **base_result}
    except Exception as exc:
        logger.warning("[generate_answer] 生成失败: %s → fallback", exc)
        langfuse_context.update_current_observation(
            node="generate_answer",
            generation_mode="non_stream",
            error=str(exc),
        )

        snippets = "\n---\n".join(
            f"[{_doc_source(d)}] {d.get('content', '')[:200]}"
            for d in selected_docs[:5]
        )
        return {
            "answer": (
                "生成服务暂时不可用。以下是检索到的相关文档片段：\n\n"
                f"{snippets}\n\n"
                f"[错误: {exc}]"
            ),
            "errors": [f"generate_answer: {exc}"],
            **base_result,
        }
