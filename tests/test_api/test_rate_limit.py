"""速率限制测试 — Redis sliding window 限流."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestRateLimit:
    """速率限制中间件测试."""

    @pytest.fixture
    def rate_limit_app(self, test_app, mock_llm_client, mock_graph):
        """配置带 Redis mock 的 app，模拟速率限制触发."""
        from src.main import app

        # 构造 mock Redis pipeline
        # Pipeline methods (zremrangebyscore, zcard, etc.) are queued NOT awaited,
        # so they must be regular MagicMock (NOT AsyncMock)
        mock_pipeline = MagicMock()
        mock_pipeline.zremrangebyscore = MagicMock()
        mock_pipeline.zcard = MagicMock()
        mock_pipeline.zadd = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock(
            return_value=[None, 31]  # zcard=31 > 30 limit → trigger 429
        )
        # async context manager support
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=None)

        mock_redis = MagicMock()
        mock_redis.pipeline = MagicMock(return_value=mock_pipeline)

        # 设置 cache_store
        mock_cache_store = MagicMock()
        mock_cache_store._redis = mock_redis
        app.state.cache_store = mock_cache_store
        app.state.llm_client = mock_llm_client
        app.state.graph = mock_graph

        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_rate_limit_exceeded_429(self, rate_limit_app):
        """超过限制返回 429."""
        resp = rate_limit_app.post("/chat", json={"query": "test"})
        assert resp.status_code == 429
        data = resp.json()
        assert "Too many requests" in data["detail"]
        assert "Retry-After" in resp.headers

    def test_health_not_rate_limited(self, test_app):
        """/health 不受速率限制."""
        resp = test_app.get("/health")
        assert resp.status_code != 429

    def test_rate_limit_fail_open(self, test_app, mock_llm_client, mock_graph):
        """Redis 不可用时 fail-open (放行)."""
        from src.main import app

        # 设置有问题的 Redis（每次操作抛异常）
        bad_redis = AsyncMock()
        bad_redis.zremrangebyscore = MagicMock(side_effect=ConnectionError("Redis unavailable"))
        mock_cache = MagicMock()
        mock_cache._redis = bad_redis
        app.state.cache_store = mock_cache
        app.state.llm_client = mock_llm_client
        app.state.graph = mock_graph

        from fastapi.testclient import TestClient
        client = TestClient(app)

        # 应该 fail-open
        resp = client.post("/chat", json={"query": "test"})
        assert resp.status_code == 200  # 请求被放行
