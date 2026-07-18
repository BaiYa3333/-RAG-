"""MCP 工具往返集成测试 — in-memory client/server session.

覆盖: tools/list 发现、工具调用往返、参数透传、异常 → tool error (isError)
且 server 存活。service 层以 monkeypatch 替身隔离（业务逻辑在 test_service.py）。
"""

import json
from unittest.mock import AsyncMock

from mcp.shared.memory import create_connected_server_and_client_session

import src.mcp_server.service as svc
from src.mcp_server.server import create_mcp_server


def _result_payload(result) -> dict:
    """从 CallToolResult 提取结构化返回（structuredContent 优先，回退 text）."""
    if getattr(result, "structuredContent", None):
        payload = result.structuredContent
        # FastMCP 对 dict 返回值可能包裹为 {"result": ...}
        return payload.get("result", payload) if isinstance(payload, dict) else payload
    return json.loads(result.content[0].text)


async def test_tools_list_exposes_three_tools_with_llm_guidance():
    mcp = create_mcp_server()
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        result = await session.list_tools()

    tools = {t.name: t for t in result.tools}
    assert set(tools) == {"rag_query", "rag_search", "list_knowledge_bases"}

    # spec mcp-tools: LLM-oriented descriptions — 说明取舍与工作流引导
    assert "rag_search" in tools["rag_query"].description
    assert "rag_query" in tools["rag_search"].description
    assert "list_knowledge_bases" in tools["rag_search"].description
    for tool in tools.values():
        assert tool.description and tool.inputSchema


async def test_rag_search_roundtrip(monkeypatch):
    expected = {
        "chunks": [
            {
                "content": "RRF 融合算法说明",
                "score": 0.91,
                "metadata": {"source": "rag.md", "kb_id": "kb-1", "doc_id": "d1", "chunk_id": "c1"},
            }
        ],
        "total": 1,
    }
    search_mock = AsyncMock(return_value=expected)
    monkeypatch.setattr(svc, "search_chunks", search_mock)

    mcp = create_mcp_server()
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        result = await session.call_tool("rag_search", {"query": "什么是 RRF", "top_k": 1})

    assert not result.isError
    assert _result_payload(result) == expected
    search_mock.assert_awaited_once_with("什么是 RRF", kb_ids=None, top_k=1)


async def test_rag_query_passes_arguments_through(monkeypatch):
    answer_mock = AsyncMock(return_value={
        "answer": "42", "citations": {}, "sources": [], "intent": "factoid", "retrieval_tier": 1,
    })
    monkeypatch.setattr(svc, "answer_question", answer_mock)

    mcp = create_mcp_server()
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        result = await session.call_tool(
            "rag_query",
            {"question": "什么是 RRF?", "kb_ids": ["kb-1"], "model": "deepseek-chat"},
        )

    assert not result.isError
    assert _result_payload(result)["answer"] == "42"
    answer_mock.assert_awaited_once_with(
        "什么是 RRF?", kb_ids=["kb-1"], model="deepseek-chat", language=None
    )


async def test_tool_error_is_readable_and_server_survives(monkeypatch):
    monkeypatch.setattr(
        svc, "list_kbs",
        AsyncMock(side_effect=ValueError("知识库服务不可用（PostgreSQL 未连接）")),
    )
    monkeypatch.setattr(
        svc, "search_chunks",
        AsyncMock(return_value={"chunks": [], "total": 0}),
    )

    mcp = create_mcp_server()
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        # 异常 → MCP tool error（isError=true + 可读消息），非协议级崩溃
        error_result = await session.call_tool("list_knowledge_bases", {})
        assert error_result.isError
        assert "知识库服务不可用" in error_result.content[0].text

        # spec mcp-server: 同一 server 后续调用不受影响
        ok_result = await session.call_tool("rag_search", {"query": "still alive"})
        assert not ok_result.isError


async def test_unexpected_exception_wrapped_readably(monkeypatch):
    monkeypatch.setattr(
        svc, "search_chunks",
        AsyncMock(side_effect=RuntimeError("chromadb connection refused")),
    )

    mcp = create_mcp_server()
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        result = await session.call_tool("rag_search", {"query": "q"})

    assert result.isError
    text = result.content[0].text
    assert "rag_search 执行失败" in text
