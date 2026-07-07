"""API 请求/响应 Pydantic Schema — 与内部 RAGState 解耦."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat 和 /chat/stream 的请求体."""

    query: str = Field(..., min_length=1, description="用户查询")
    chat_history: list[dict[str, str]] | None = Field(
        default=None, description="多轮对话历史 [{role, content}]"
    )
    session_id: str | None = Field(default=None, description="持久会话 ID")
    language: str | None = Field(default=None, description="输出语言覆盖，如 zh/en")
    model: str | None = Field(default=None, description="本次请求使用的 LLM 模型名")
    kb_ids: list[str] | None = Field(default=None, description="目标知识库 ID 列表")
    user_id: str | None = Field(default=None, description="用户 ID（由认证系统填充）")


class DocumentSchema(BaseModel):
    """返回给客户端的文档格式."""

    content: str = Field(..., description="文档内容")
    score: float = Field(..., description="相关性分数")
    metadata: dict[str, Any] = Field(default_factory=dict, description="文档元数据")


class GateLogEntry(BaseModel):
    """质量门控日志条目."""

    tier: int
    action: str  # pass | escalate | fallback_generate
    avg_score: float | None = None
    reason: str | None = None


class ChatResponse(BaseModel):
    """POST /chat 的非流式响应."""

    query: str = Field(..., description="原始查询")
    answer: str = Field(..., description="生成的回答")
    intent: str = Field(..., description="意图分类: factoid | comparison | summary | analytical")
    documents: list[DocumentSchema] = Field(default_factory=list)
    retrieval_tier: int = Field(..., description="最终检索层级: 1 或 2")
    gate_log: list[GateLogEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    model: str = Field(..., description="实际使用的 LLM 模型名")
    session_id: str | None = Field(default=None, description="持久会话 ID")
    citations: dict[str, dict] = Field(default_factory=dict, description="文档引用映射 {ref_num: {source, page, snippet, score}}")


class ChatStreamEvent(BaseModel):
    """SSE 事件负载的基类."""

    event: str = Field(..., description="事件类型: intent | documents | token | done | error")
    data: dict[str, Any] = Field(..., description="事件数据")


class UploadResponse(BaseModel):
    """POST /documents/upload 的响应."""

    doc_id: str | None = Field(None, description="PostgreSQL 文档记录 UUID（DocStore 不可用时为 None）")
    title: str = Field(..., description="文件名")
    source: str = Field(..., description="上传时的临时文件路径")
    doc_type: str = Field(..., description="文件扩展名（小写）")
    chunk_count: int = Field(..., description="切分产生的子块数量")
    status: str = Field(..., description="摄入状态: indexed | partial")
    file_hash: str = Field(..., description="文件内容 SHA-256 哈希")
    metadata: dict[str, Any] = Field(default_factory=dict, description="文档级元数据")


# ── 评估 (RAGAS) Schema ──────────────────────────────────────


class TestsetItem(BaseModel):
    """单条 RAGAS 测试用例."""

    question: str = Field(..., min_length=1, description="测试问题")
    ground_truth: str = Field(..., min_length=1, description="参考答案")
    reference_contexts: list[str] | None = Field(
        default=None, description="参考上下文列表"
    )


class EvalRequest(BaseModel):
    """POST /eval/run 的请求体."""

    testset_source: str = Field(
        default="auto",
        pattern="^(auto|manual)$",
        description="测试集来源: auto=自动生成, manual=手动提供",
    )
    document_ids: list[str] | None = Field(
        default=None, description="auto 模式下指定的文档 ID 列表"
    )
    testset: list[TestsetItem] | None = Field(
        default=None, description="manual 模式下用户提供的测试集"
    )
    testset_size: int = Field(
        default=10, ge=1, le=100, description="auto 模式下生成的测试集大小"
    )
    kb_ids: list[str] | None = Field(
        default=None, description="目标知识库 ID 列表（为空时搜索默认集合）"
    )


class EvalResponse(BaseModel):
    """POST /eval/run 的响应."""

    evaluation_id: str = Field(..., description="评估 UUID v4")
    faithfulness: float = Field(..., description="忠实度 (0.0-1.0)")
    answer_relevancy: float = Field(..., description="答案相关性 (0.0-1.0)")
    context_precision: float = Field(..., description="上下文精度 (0.0-1.0)")
    context_recall: float = Field(..., description="上下文召回率 (0.0-1.0)")
    avg_score: float = Field(..., description="四项指标平均分")
    testset_size: int = Field(..., description="测试集大小")
    created_at: str = Field(..., description="评估时间 (ISO 8601)")


class EvalResultsResponse(BaseModel):
    """GET /eval/results 的响应."""

    total: int = Field(..., description="评估历史总数")
    results: list[EvalResponse] = Field(default_factory=list, description="评估结果列表")


# ── Session & Memory Schemas ──────────────────────────────────


class MemoryItemResponse(BaseModel):
    """GET /memory 的响应项."""

    id: str
    user_id: str
    memory_type: str
    content: str
    source_session_id: str | None = Field(default=None)
    created_at: str | None = Field(default=None)
    expires_at: str | None = Field(default=None)


class CreateSessionRequest(BaseModel):
    """POST /sessions 的请求体."""

    title: str | None = Field(default=None)


class SessionResponse(BaseModel):
    """会话列表项响应."""

    id: str
    user_id: str | None = None
    title: str | None = None
    summary: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class SessionDetailResponse(BaseModel):
    """GET /sessions/{id} 的详细响应."""

    id: str
    user_id: str | None = None
    title: str | None = None
    summary: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    history: list[dict] = []
