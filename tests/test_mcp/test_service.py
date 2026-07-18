"""MCP 服务层单元测试 — RAGState 映射 / metadata 白名单 / 权限预检 / 超时."""

import asyncio
from unittest.mock import AsyncMock

import pytest

import src.mcp_server.service as svc

KB_LIST = [
    {
        "id": "kb-1",
        "name": "product-docs",
        "display_name": "产品文档",
        "description": "产品手册与 FAQ",
        "doc_count": 42,
        "is_public": True,
    },
    {
        "id": "kb-2",
        "name": "eng-wiki",
        "display_name": None,
        "description": None,
        "doc_count": 7,
        "is_public": False,
    },
]

RAW_DOC = {
    "content": "RRF 是 Reciprocal Rank Fusion 的缩写。",
    "score": 0.87,
    "rrf_score": 0.032,
    "metadata": {
        "source": "rag.md",
        "kb_id": "kb-1",
        "doc_id": "doc-9",
        "chunk_id": "abc_p0001_c0002_deadbeef",
        # 内部字段 — 白名单必须剥离
        "parent_content": "x" * 1024,
        "parent_chunk_id": "abc_p0001",
        "kw_match_score": 0.5,
    },
}


@pytest.fixture(autouse=True)
def mcp_service_state(monkeypatch):
    """注入 mock KB 服务并在测试后重置模块级单例."""
    kb_service = AsyncMock()
    kb_service.list_kbs = AsyncMock(return_value=[dict(kb) for kb in KB_LIST])
    svc.set_kb_service(kb_service)
    yield kb_service
    svc.set_kb_service(None)
    svc._graph = None
    svc._own_doc_store = None


# ── answer_question ──────────────────────────────────────────


async def test_answer_question_maps_state_and_slims_metadata(monkeypatch):
    state = {
        "answer": "RRF 是一种结果融合算法。",
        "citations": {"1": "rag.md"},
        "documents": [dict(RAW_DOC)],
        "intent": "factoid",
        "retrieval_tier": 2,
        "errors": [],
    }
    run_query_mock = AsyncMock(return_value=state)
    monkeypatch.setattr(svc, "run_query", run_query_mock)
    monkeypatch.setattr(svc, "_get_graph", AsyncMock(return_value=None))

    result = await svc.answer_question("什么是 RRF?", kb_ids=["kb-1"])

    assert result["answer"] == state["answer"]
    assert result["intent"] == "factoid"
    assert result["retrieval_tier"] == 2
    assert result["citations"] == {"1": "rag.md"}
    assert len(result["sources"]) == 1
    src_doc = result["sources"][0]
    assert src_doc["content"] == RAW_DOC["content"]
    assert src_doc["score"] == 0.87
    # Decision 10: 白名单 — 内部字段绝不外流
    assert set(src_doc["metadata"]) <= {"source", "kb_id", "doc_id", "chunk_id"}
    assert "parent_content" not in src_doc["metadata"]
    assert "parent_chunk_id" not in src_doc["metadata"]

    # Decision 5: 服务身份贯穿（mcp_user_id 默认空 → None）
    _, kwargs = run_query_mock.await_args
    assert kwargs["user_id"] is None
    assert kwargs["kb_ids"] == ["kb-1"]


async def test_answer_question_rejects_unknown_model():
    with pytest.raises(ValueError, match="未知模型"):
        await svc.answer_question("q", model="not-a-model")


async def test_answer_question_rejects_empty_question():
    with pytest.raises(ValueError, match="question 不能为空"):
        await svc.answer_question("   ")


async def test_answer_question_resolves_all_accessible_kbs(monkeypatch):
    run_query_mock = AsyncMock(return_value={"answer": "", "documents": []})
    monkeypatch.setattr(svc, "run_query", run_query_mock)
    monkeypatch.setattr(svc, "_get_graph", AsyncMock(return_value=None))

    await svc.answer_question("q")  # kb_ids 省略 → 全部可访问 KB

    _, kwargs = run_query_mock.await_args
    assert kwargs["kb_ids"] == ["kb-1", "kb-2"]


# ── 权限预检 ─────────────────────────────────────────────────


async def test_unauthorized_kb_id_rejected_with_guidance():
    with pytest.raises(ValueError) as exc_info:
        await svc.search_chunks("q", kb_ids=["kb-1", "kb-nope"])
    msg = str(exc_info.value)
    assert "kb-nope" in msg
    assert "list_knowledge_bases" in msg


async def test_kb_precheck_skipped_when_kb_service_down(monkeypatch):
    monkeypatch.setattr(svc, "_get_kb_service", AsyncMock(return_value=None))
    monkeypatch.setattr(svc, "_embed_query", AsyncMock(return_value=[0.1] * 1024))
    monkeypatch.setattr(svc, "dense_search", AsyncMock(return_value=[]))
    monkeypatch.setattr(svc, "_ensure_sparse_indexes", AsyncMock(return_value=[]))

    # 降级：显式 kb_ids 原样放行，不抛错
    result = await svc.search_chunks("q", kb_ids=["kb-any"])
    assert result == {"chunks": [], "total": 0}


# ── search_chunks ────────────────────────────────────────────


def _patch_search_primitives(monkeypatch, dense_docs, rerank_passthrough=True):
    monkeypatch.setattr(svc, "_embed_query", AsyncMock(return_value=[0.1] * 1024))
    monkeypatch.setattr(svc, "dense_search", AsyncMock(return_value=dense_docs))
    monkeypatch.setattr(svc, "_ensure_sparse_indexes", AsyncMock(return_value=[]))
    if rerank_passthrough:
        async def _fake_rerank(query, documents, top_k=None, **kwargs):
            return documents[:top_k]
        monkeypatch.setattr(svc, "rerank", _fake_rerank)


async def test_search_chunks_truncates_to_top_k_and_slims(monkeypatch):
    # 每条独立 parent_chunk_id — RRF 按 parent 去重，共享 parent 会合并
    docs = [
        {
            **RAW_DOC,
            "content": f"doc-{i}",
            "metadata": {**RAW_DOC["metadata"], "chunk_id": f"c{i}", "parent_chunk_id": f"p{i}"},
        }
        for i in range(10)
    ]
    _patch_search_primitives(monkeypatch, docs)

    result = await svc.search_chunks("什么是 RRF", top_k=3)

    assert result["total"] == 3
    assert len(result["chunks"]) == 3
    for chunk in result["chunks"]:
        assert set(chunk) == {"content", "score", "metadata"}
        assert "parent_content" not in chunk["metadata"]


async def test_search_chunks_empty_result_is_success(monkeypatch):
    _patch_search_primitives(monkeypatch, [])
    result = await svc.search_chunks("冷门查询")
    assert result == {"chunks": [], "total": 0}


async def test_search_chunks_timeout_yields_readable_error(monkeypatch):
    async def _slow_embed(query):
        await asyncio.sleep(1.0)
        return [0.1] * 1024

    monkeypatch.setattr(svc, "_embed_query", _slow_embed)
    monkeypatch.setattr(svc, "SEARCH_TIMEOUT_S", 0.05)

    with pytest.raises(ValueError, match="检索超时"):
        await svc.search_chunks("q")


async def test_search_chunks_rejects_empty_query():
    with pytest.raises(ValueError, match="query 不能为空"):
        await svc.search_chunks("")


async def test_search_chunks_clamps_top_k(monkeypatch):
    docs = [
        {
            **RAW_DOC,
            "content": f"doc-{i}",
            "metadata": {**RAW_DOC["metadata"], "parent_chunk_id": f"p{i}"},
        }
        for i in range(50)
    ]
    _patch_search_primitives(monkeypatch, docs)

    result = await svc.search_chunks("q", top_k=999)
    assert result["total"] == svc.MAX_TOP_K


# ── list_kbs ─────────────────────────────────────────────────


async def test_list_kbs_maps_fields():
    result = await svc.list_kbs()
    kbs = result["knowledge_bases"]
    assert len(kbs) == 2
    assert kbs[0] == {
        "id": "kb-1",
        "name": "产品文档",
        "description": "产品手册与 FAQ",
        "document_count": 42,
    }
    # display_name 为空回退 name；description 为空回退空串
    assert kbs[1]["name"] == "eng-wiki"
    assert kbs[1]["description"] == ""


async def test_list_kbs_unavailable_raises_readable_error(monkeypatch):
    monkeypatch.setattr(svc, "_get_kb_service", AsyncMock(return_value=None))
    with pytest.raises(ValueError, match="知识库服务不可用"):
        await svc.list_kbs()


async def test_list_kbs_uses_service_identity(monkeypatch, mcp_service_state):
    monkeypatch.setattr(svc.settings, "mcp_user_id", "svc-user-1")
    await svc.list_kbs()
    mcp_service_state.list_kbs.assert_awaited_with(user_id="svc-user-1")


# ── _get_kb_service 惰性连接 ─────────────────────────────────


async def test_get_kb_service_retries_after_failed_connect(monkeypatch):
    """连接失败不得毒化 _own_doc_store 单例 — 后端恢复后重试必须成功."""
    svc.set_kb_service(None)

    connect_calls = {"n": 0}

    class FakeDocStore:
        async def connect(self):
            connect_calls["n"] += 1
            if connect_calls["n"] == 1:
                raise ConnectionError("PostgreSQL down")

    monkeypatch.setattr("src.stores.DocStore", FakeDocStore)

    # 首次: PG 未就绪 → 降级返回 None，且不得留下未连接实例
    assert await svc._get_kb_service() is None
    assert svc._own_doc_store is None

    # 后端恢复: 重试必须真正重新 connect 并成功
    kb_service = await svc._get_kb_service()
    assert kb_service is not None
    assert connect_calls["n"] == 2
