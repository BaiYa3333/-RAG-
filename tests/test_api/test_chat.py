"""POST /chat 端点测试 — 非流式 RAG 查询."""


class TestChatEndpoint:
    """非流式 /chat 端点测试."""

    def test_chat_success(self, test_app):
        """正常请求返回完整 RAG 响应."""
        resp = test_app.post("/chat", json={"query": "什么是 RRF"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "什么是 RRF"
        assert len(data["answer"]) > 0
        assert data["intent"] == "factoid"
        assert len(data["documents"]) > 0
        assert data["retrieval_tier"] == 1
        assert len(data["gate_log"]) > 0

    def test_chat_with_chat_history(self, test_app):
        """带 chat_history 的多轮查询."""
        resp = test_app.post(
            "/chat",
            json={
                "query": "那它的应用呢",
                "chat_history": [
                    {"role": "user", "content": "什么是 RRF"},
                    {"role": "assistant", "content": "RRF 是 Reciprocal Rank Fusion。"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # mock graph returns fixed state — query field is from mock, not the input
        assert "query" in data
        assert len(data["answer"]) > 0

    def test_chat_empty_query_422(self, test_app):
        """空 query 返回 422."""
        resp = test_app.post("/chat", json={"query": ""})
        assert resp.status_code == 422

    def test_chat_missing_query_422(self, test_app):
        """缺失 query 字段返回 422."""
        resp = test_app.post("/chat", json={})
        assert resp.status_code == 422
        # 错误信息应该包含 'query'
        assert "query" in str(resp.json()).lower() or "query" in resp.text.lower()

    def test_chat_response_schema_valid(self, test_app):
        """验证响应 Schema 完整性."""
        resp = test_app.post("/chat", json={"query": "什么是 RRF"})
        assert resp.status_code == 200
        data = resp.json()

        # 检查所有必填字段存在
        required_fields = ["query", "answer", "intent", "documents", "retrieval_tier", "gate_log", "errors"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        # documents 结构
        if data["documents"]:
            doc = data["documents"][0]
            assert "content" in doc
            assert "score" in doc
            assert "metadata" in doc

        # gate_log 结构
        if data["gate_log"]:
            entry = data["gate_log"][0]
            assert "tier" in entry
            assert "action" in entry
