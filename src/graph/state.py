"""LangGraph RAGState — 工作流状态 Schema (LangGraph 1.0)."""

from typing import Annotated, Any, TypedDict
from operator import add


class RAGState(TypedDict):
    """RAG 工作流全局状态。

    Annotated 字段使用 `operator.add` reducer 累积跨节点数据。
    """

    # ── 输入 ──
    query: str
    chat_history: list[dict]
    session_id: str | None
    user_id: str | None
    language: str | None
    model_name: str
    stream_mode: bool  # True when token-level streaming should be used

    # ── 知识库 ──
    kb_ids: list[str]  # 目标知识库列表

    # ── 路由 ──
    intent: str  # factoid | comparison | summary | analytical
    suggested_k: int
    suggested_tools: list[str]
    router_confidence: float

    # ── 检索 ──
    documents: list[dict]
    retrieval_scores: list[float]
    retrieval_tier: int  # 1 or 2

    # ── 质量门控 ──
    gate_log: Annotated[list[dict], add]

    # ── Agent (analytical only) ──
    agent_iterations: int
    agent_sub_queries: list[str]
    agent_fallback: bool
    tool_calls: list[dict]
    tool_results: list[dict]

    # ── 重排序 ──
    rerank_applied: bool
    rerank_method: str  # api | llm | identity | none

    # ── 生成 ──
    compressed_context: str   # 压缩后的上下文（summary/analytical 专用）
    used_contexts: list[str]  # 实际送入 LLM / RAGAS 的上下文
    used_context_count: int
    used_sources: list[str]
    answer: str
    citations: dict[str, dict]  # 引用映射 {"1": {source, page, snippet, score}}
    streaming_context: str    # 流式模式下的 LLM 上下文（仅流式模式使用）
    streaming_intent: str     # 流式模式下的意图（仅流式模式使用）

    # ── 消息 & 错误 ──
    messages: Annotated[list[dict], add]
    errors: Annotated[list[str], add]


def initial_state(
    query: str,
    chat_history: list[dict] | None = None,
    model_name: str | None = None,
    session_id: str | None = None,
    language: str | None = None,
    user_id: str | None = None,
    kb_ids: list[str] | None = None,
    stream_mode: bool = False,
) -> RAGState:
    """构建初始状态。"""

    from src.llm.registry import DEFAULT_MODEL

    return RAGState(
        query=query,
        chat_history=chat_history or [],
        session_id=session_id,
        user_id=user_id,
        language=language,
        model_name=model_name or DEFAULT_MODEL,
        kb_ids=kb_ids or [],
        stream_mode=stream_mode,
        intent="",
        suggested_k=5,
        suggested_tools=["dense_search"],
        router_confidence=0.0,
        documents=[],
        retrieval_scores=[],
        retrieval_tier=1,
        gate_log=[],
        agent_iterations=0,
        agent_sub_queries=[],
        agent_fallback=False,
        rerank_applied=False,
        rerank_method="none",
        compressed_context="",
        used_contexts=[],
        used_context_count=0,
        used_sources=[],
        answer="",
        citations={},
        streaming_context="",
        streaming_intent="",
        tool_calls=[],
        tool_results=[],
        messages=[],
        errors=[],
    )
