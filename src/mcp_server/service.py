"""MCP 服务层 — 传输无关的 RAG 能力封装.

被 stdio 与 Streamable HTTP 两种传输共享，MCP 层零业务逻辑复制：
- answer_question 委托 run_query()（完整管道，含 60s 超时与 graceful degradation）
- search_chunks 组合既有检索原语（embed → dense+BM25 → RRF → rerank），30s 超时
- list_kbs 委托 KnowledgeBaseService，既有 KB 权限模型自然生效

所有方法以 RAG_MCP_USER_ID 服务身份执行（空 = 匿名，仅可见 public KB）。

设计约束（openspec/changes/add-mcp-server/design.md）:
- Decision 4: 复用管道入口与检索原语，不构造伪 RAGState
- Decision 9: 无状态 — 不接受 session_id / chat_history，每次调用独立成立
- Decision 10: 返回结构 metadata 白名单，剥离 parent_content 等内部字段
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.config import settings

# 复用 graph 检索节点的模块级单例（embedder + per-collection BM25 索引缓存），
# 避免 MCP 路径重复构建索引 / 与 warmup、invalidate 生命周期脱节。
from src.graph.nodes.retrieval import _embed_query, _ensure_sparse_indexes
from src.graph.workflow import build_workflow, run_query
from src.observability.decorators import trace_rag_node
from src.rag.retrieval.dense import dense_search
from src.rag.retrieval.reranker import rerank
from src.rag.retrieval.rrf import reciprocal_rank_fusion
from src.utils.logger import logger

# rag_search 独立超时（秒）— 不走 LangGraph，无全局 60s 保护
SEARCH_TIMEOUT_S = 30.0

# rag_search 召回宽度（rerank 前），对齐 tier2 的默认值
SEARCH_RECALL_K = 20

# rag_search top_k 上限 — 每条 chunk 为 parent 级文本（~1k token），
# 上限防止单次调用向 Host 上下文注入过多内容
MAX_TOP_K = 20

# Decision 10: chunk metadata 白名单 — 仅保留溯源必需字段。
# 白名单而非黑名单：未来 metadata 新增内部字段不会无意识泄漏给外部 Host。
META_WHITELIST = ("source", "kb_id", "doc_id", "chunk_id")

# ── 依赖单例（HTTP 模式由 main.py lifespan 注入共享实例；stdio 模式惰性自建）──
_kb_service: Any = None
_own_doc_store: Any = None
_graph: Any = None
_graph_lock = asyncio.Lock()


def set_kb_service(kb_service: Any) -> None:
    """注入共享 KnowledgeBaseService（main.py lifespan 调用；传 None 重置）。"""
    global _kb_service
    _kb_service = kb_service


async def _get_kb_service() -> Any:
    """获取 KB 服务 — 未注入时惰性自建 DocStore 连接（stdio 独立进程模式）。

    连接失败返回 None（降级：调用方按「KB 服务不可用」处理）。
    """
    global _kb_service, _own_doc_store
    if _kb_service is not None:
        return _kb_service

    try:
        from src.rag.knowledge_base.service import KnowledgeBaseService
        from src.stores import DocStore

        if _own_doc_store is None:
            # 先 connect 成功再赋值单例 — 避免连接失败留下「已构造未连接」的
            # 毒化实例，导致后端恢复后所有重试永久走降级路径
            store = DocStore()
            await store.connect()
            _own_doc_store = store
        _kb_service = KnowledgeBaseService(_own_doc_store)
        return _kb_service
    except Exception as exc:
        logger.warning("mcp_kb_service_unavailable", error=str(exc))
        return None


async def _get_graph() -> Any:
    """惰性编译 LangGraph workflow 单例（避免每次 rag_query 重复 build）。"""
    global _graph
    if _graph is None:
        async with _graph_lock:
            if _graph is None:
                _graph = build_workflow()
    return _graph


def _slim_doc(doc: dict) -> dict:
    """将管道内部 document dict 映射为 MCP 返回结构（Decision 10 白名单）。"""
    meta = doc.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    score = doc.get("score", doc.get("rrf_score", 0.0)) or 0.0
    return {
        "content": doc.get("content", ""),
        "score": round(float(score), 4),
        "metadata": {k: meta[k] for k in META_WHITELIST if meta.get(k) is not None},
    }


def _validate_model(model: str | None) -> None:
    """模型名校验 — 无效时抛出列出可用模型的可读错误（供 Host LLM 自纠正）。"""
    if not model:
        return
    from src.llm.registry import MODEL_REGISTRY

    if model not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"未知模型 '{model}'。可用模型: {available}。省略 model 参数将使用默认模型。"
        )


async def _accessible_kbs() -> list[dict] | None:
    """列出服务身份可访问的知识库；KB 服务不可用时返回 None（降级）。"""
    kb_service = await _get_kb_service()
    if kb_service is None:
        return None
    try:
        return await kb_service.list_kbs(user_id=settings.mcp_user_id or None)
    except Exception as exc:
        logger.warning("mcp_list_kbs_failed", error=str(exc))
        return None


async def _resolve_kb_ids(kb_ids: list[str] | None) -> list[str] | None:
    """解析检索范围 + 权限预检。

    - kb_ids 为空 → 服务身份可访问的全部 KB（避免回退到空的遗留默认集合）
    - kb_ids 显式给出 → 逐一校验可访问性，含不可访问项时抛可读错误
    - KB 服务不可用（PG 未连接）→ 显式 kb_ids 原样放行（降级，检索层对
      不存在的 collection 自然返回空），为空则返回 None 回退默认集合
    """
    accessible = await _accessible_kbs()

    if kb_ids:
        if accessible is None:
            logger.warning("mcp_kb_precheck_skipped", kb_ids=kb_ids)
            return kb_ids
        allowed = {kb["id"] for kb in accessible}
        denied = [kid for kid in kb_ids if kid not in allowed]
        if denied:
            raise ValueError(
                f"kb_id {denied} 不存在或无权访问。"
                "请先调用 list_knowledge_bases 获取有效的 kb_ids 后重试。"
            )
        return kb_ids

    if accessible:
        return [kb["id"] for kb in accessible]
    return None


# ── 工具实现 ─────────────────────────────────────────────────


@trace_rag_node(name="mcp_rag_query")
async def answer_question(
    question: str,
    kb_ids: list[str] | None = None,
    model: str | None = None,
    language: str | None = None,
) -> dict:
    """完整 RAG 问答 — 委托 run_query()，映射 RAGState 为 MCP 工具输出。"""
    if not question or not question.strip():
        raise ValueError("question 不能为空。请传入自包含的完整问题。")
    _validate_model(model)

    start = time.perf_counter()
    effective_kb_ids = await _resolve_kb_ids(kb_ids)
    graph = await _get_graph()

    state = await run_query(
        question.strip(),
        graph=graph,
        model=model,
        language=language,
        user_id=settings.mcp_user_id or None,
        kb_ids=effective_kb_ids,
    )

    result = {
        "answer": state.get("answer", ""),
        "citations": state.get("citations") or {},
        "sources": [_slim_doc(d) for d in (state.get("documents") or [])],
        "intent": state.get("intent", ""),
        "retrieval_tier": state.get("retrieval_tier", 1),
    }
    logger.info(
        "mcp_rag_query_done",
        transport="mcp",
        kb_count=len(effective_kb_ids or []),
        intent=result["intent"],
        tier=result["retrieval_tier"],
        source_count=len(result["sources"]),
        answer_chars=len(result["answer"]),
        elapsed_ms=round((time.perf_counter() - start) * 1000),
        errors=len(state.get("errors") or []),
    )
    return result


@trace_rag_node(name="mcp_rag_search")
async def search_chunks(
    query: str,
    kb_ids: list[str] | None = None,
    top_k: int | None = None,
) -> dict:
    """仅检索 — dense + BM25 → RRF → rerank，返回白名单化的 chunk 列表。"""
    if not query or not query.strip():
        raise ValueError("query 不能为空。")
    k = max(1, min(top_k or settings.mcp_default_top_k, MAX_TOP_K))

    start = time.perf_counter()
    effective_kb_ids = await _resolve_kb_ids(kb_ids)

    try:
        chunks = await asyncio.wait_for(
            _search_impl(query.strip(), effective_kb_ids, k),
            timeout=SEARCH_TIMEOUT_S,
        )
    except TimeoutError:
        raise ValueError(
            f"检索超时（>{SEARCH_TIMEOUT_S:.0f}s），请缩小检索范围（指定 kb_ids）后重试。"
        )

    logger.info(
        "mcp_rag_search_done",
        transport="mcp",
        kb_count=len(effective_kb_ids or []),
        top_k=k,
        chunk_count=len(chunks),
        elapsed_ms=round((time.perf_counter() - start) * 1000),
    )
    return {"chunks": chunks, "total": len(chunks)}


async def _search_impl(query: str, kb_ids: list[str] | None, top_k: int) -> list[dict]:
    """混合检索实现 — 镜像 tier2 的多 collection 编排（不含 HyDE / 质量门控）。"""
    embedding = await _embed_query(query)
    if not embedding:
        raise ValueError("查询 embedding 生成失败，请稍后重试。")

    dense_results = await dense_search(embedding, top_k=SEARCH_RECALL_K, kb_ids=kb_ids)

    sparse_results: list[dict] = []
    seen_sparse_ids: set[str] = set()
    try:
        sparse_pairs = await _ensure_sparse_indexes(kb_ids=kb_ids)
        for _col_name, sp in sparse_pairs:
            col_results = await asyncio.get_event_loop().run_in_executor(
                None, sp.search, query, SEARCH_RECALL_K,
            )
            for doc in col_results:
                key = doc.get("id", "")
                if key and key not in seen_sparse_ids:
                    seen_sparse_ids.add(key)
                    sparse_results.append(doc)
    except Exception as exc:
        # BM25 失败降级为纯 dense（与 tier2 的容错哲学一致）
        logger.warning("mcp_sparse_search_failed", error=str(exc))

    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    reranked = await rerank(query, fused, top_k=top_k)
    return [_slim_doc(d) for d in reranked]


@trace_rag_node(name="mcp_list_kbs")
async def list_kbs() -> dict:
    """列出服务身份可访问的知识库（id / name / description / document_count）。"""
    start = time.perf_counter()
    kbs = await _accessible_kbs()
    if kbs is None:
        raise ValueError(
            "知识库服务不可用（PostgreSQL 未连接）。"
            "可省略 kb_ids 直接调用 rag_search / rag_query 检索默认集合。"
        )

    result = {
        "knowledge_bases": [
            {
                "id": kb["id"],
                "name": kb.get("display_name") or kb.get("name") or "",
                "description": kb.get("description") or "",
                "document_count": kb.get("doc_count", 0),
            }
            for kb in kbs
        ]
    }
    logger.info(
        "mcp_list_kbs_done",
        transport="mcp",
        kb_count=len(result["knowledge_bases"]),
        elapsed_ms=round((time.perf_counter() - start) * 1000),
    )
    return result
