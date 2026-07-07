"""POST /chat/stream 端点测试 — SSE 流式 RAG 查询."""

import json


class TestChatStreamEndpoint:
    """流式 /chat/stream 端点测试."""

    def test_stream_content_type_is_sse(self, test_app):
        """响应 Content-Type 为 text/event-stream."""
        resp = test_app.post("/chat/stream", json={"query": "什么是 RRF"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    def test_stream_cache_headers(self, test_app):
        """响应包含 SSE 缓存控制头."""
        resp = test_app.post("/chat/stream", json={"query": "什么是 RRF"})
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["connection"] == "keep-alive"

    def test_stream_emits_intent_event(self, test_app):
        """SSE 流包含 intent 事件."""
        resp = test_app.post("/chat/stream", json={"query": "什么是 RRF"})
        body = resp.text

        assert "event: intent" in body
        # 验证 JSON 数据可解析
        data_line = _extract_event_data(body, "intent")
        assert data_line is not None
        data = json.loads(data_line)
        assert "intent" in data

    def test_stream_emits_documents_event(self, test_app):
        """SSE 流包含 documents 事件."""
        resp = test_app.post("/chat/stream", json={"query": "什么是 RRF"})
        body = resp.text

        assert "event: documents" in body

    def test_stream_emits_done_event(self, test_app):
        """SSE 流以 done 事件结束."""
        resp = test_app.post("/chat/stream", json={"query": "什么是 RRF"})
        body = resp.text

        assert "event: done" in body

    def test_stream_empty_query_422(self, test_app):
        """空 query 返回 422."""
        resp = test_app.post("/chat/stream", json={"query": ""})
        assert resp.status_code == 422


def _extract_event_data(body: str, event_type: str) -> str | None:
    """从 SSE body 中提取指定事件类型的 data 行."""
    lines = body.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == f"event: {event_type}":
            # Next line should be data: ...
            if i + 1 < len(lines):
                data_line = lines[i + 1].strip()
                if data_line.startswith("data:"):
                    return data_line[5:].strip()
    return None
