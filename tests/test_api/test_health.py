"""/health 端点测试 — 验证扩展后的健康检查."""


class TestHealthEndpoint:
    """健康检查端点测试."""

    def test_health_returns_200(self, test_app, mock_llm_client, mock_graph):
        """健康检查返回 200 状态."""
        from src.main import app

        # 配置 mock 状态
        app.state.llm_client = mock_llm_client
        app.state.graph = mock_graph

        from fastapi.testclient import TestClient
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_health_includes_graph_status(self, test_app, mock_llm_client, mock_graph):
        """/health 包含 graph 状态."""
        from src.main import app

        app.state.llm_client = mock_llm_client
        app.state.graph = mock_graph

        from fastapi.testclient import TestClient
        client = TestClient(app)

        resp = client.get("/health")
        data = resp.json()
        assert "services" in data
        assert "graph" in data["services"]
        assert data["services"]["graph"] == "ready"

    def test_health_includes_llm_status(self, test_app, mock_llm_client, mock_graph):
        """/health 包含 llm 状态."""
        from src.main import app

        app.state.llm_client = mock_llm_client
        app.state.graph = mock_graph

        from fastapi.testclient import TestClient
        client = TestClient(app)

        resp = client.get("/health")
        data = resp.json()
        assert "llm" in data["services"]
        assert data["services"]["llm"] == "ready"

    def test_health_graph_not_initialized(self, test_app, mock_llm_client):
        """Graph 未初始化时返回 not_initialized."""
        from src.main import app

        app.state.llm_client = mock_llm_client
        app.state.graph = None

        from fastapi.testclient import TestClient
        client = TestClient(app)

        resp = client.get("/health")
        data = resp.json()
        assert data["services"]["graph"] == "not_initialized"

    def test_health_llm_not_initialized(self, test_app, mock_graph):
        """LLM 未初始化时返回 not_initialized."""
        from src.main import app

        app.state.llm_client = None
        app.state.graph = mock_graph

        from fastapi.testclient import TestClient
        client = TestClient(app)

        resp = client.get("/health")
        data = resp.json()
        assert data["services"]["llm"] == "not_initialized"

    def test_health_services_structure(self, test_app, mock_llm_client, mock_graph):
        """验证 /health 返回所有 5 个服务状态."""
        from src.main import app

        app.state.llm_client = mock_llm_client
        app.state.graph = mock_graph

        from fastapi.testclient import TestClient
        client = TestClient(app)

        resp = client.get("/health")
        data = resp.json()

        expected_services = ["postgres", "redis", "chromadb", "llm", "graph"]
        for svc in expected_services:
            assert svc in data["services"], f"Missing service: {svc}"
