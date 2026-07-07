"""pytest 配置 — 共享 fixtures 和 mock 策略."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_llm_client():
    """返回预设响应的 mock LLM 客户端."""
    client = MagicMock()
    client.model_id = "deepseek-chat"
    client.default_params = {"temperature": 0.3, "max_tokens": 2048}

    inner = AsyncMock()
    chat_response = MagicMock()
    chat_response.choices = [
        MagicMock(message=MagicMock(content="这是测试回答：RRF 是 Reciprocal Rank Fusion 的缩写。"))
    ]
    inner.chat.completions.create = AsyncMock(return_value=chat_response)
    client.client = inner

    return client


@pytest.fixture
def mock_embedding_client():
    """返回固定 1024 维向量的 mock embedding 客户端."""
    client = AsyncMock()
    emb_response = MagicMock()
    emb_response.data = [MagicMock(embedding=[0.1] * 1024)]
    client.embeddings.create = AsyncMock(return_value=emb_response)
    return client


@pytest.fixture
def mock_graph():
    """返回 mock 的编译 StateGraph，返回预设 RAGState."""
    graph = AsyncMock()
    mock_state = {
        "query": "什么是 RRF",
        "answer": "RRF 是 Reciprocal Rank Fusion 的缩写，是一种结果融合算法。",
        "intent": "factoid",
        "model_name": "deepseek-chat",
        "documents": [
            {
                "content": "RRF (Reciprocal Rank Fusion) 使用 1/(k+rank) 公式融合多个检索结果。",
                "score": 0.92,
                "metadata": {"source": "doc1.pdf", "page": 3},
            }
        ],
        "retrieval_tier": 1,
        "gate_log": [{"tier": 1, "avg_score": 0.85, "action": "pass", "reason": None}],
        "errors": [],
    }
    graph.ainvoke = AsyncMock(return_value=mock_state)
    graph.astream = MagicMock(return_value=AsyncMock())

    # Mock astream to yield states (accept **kwargs for LangGraph config param)
    async def _mock_astream(state, **kwargs):
        yield {"intent_router": {"intent": "factoid", "router_confidence": 0.95}}
        yield {"tier1_retrieve": {"documents": mock_state["documents"], "retrieval_tier": 1}}
        yield {"generate_answer": {"answer": "RRF 是 Reciprocal Rank Fusion 的缩写。"}}

    graph.astream = _mock_astream
    return graph


@pytest.fixture
def test_app(mock_llm_client, mock_graph):
    """创建带 mock 状态的 FastAPI TestClient.

    禁用 auth 以简化非 auth 测试；auth 测试使用独立的 auth_test_app fixture。
    """
    from fastapi.testclient import TestClient
    from src.main import app

    # 注入 mock 状态
    app.state.llm_client = mock_llm_client
    app.state.graph = mock_graph
    app.state.cache_store = None  # 禁用 Redis（rate limit 不测试时不影响）
    app.state.vector_store = None
    app.state.doc_store = None

    # Disable auth for non-auth tests
    import src.auth.dependencies as deps
    original_auth = deps.settings.auth_enabled
    deps.settings.auth_enabled = False

    client = TestClient(app)
    yield client

    # 清理
    deps.settings.auth_enabled = original_auth
    app.state.llm_client = None
    app.state.graph = None
