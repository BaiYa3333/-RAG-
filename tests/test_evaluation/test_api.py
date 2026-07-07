"""测试评估 API 端点."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_eval_runner():
    """返回预设评估结果的 mock EvaluationRunner."""
    runner = AsyncMock()
    runner.run = AsyncMock(return_value={
        "evaluation_id": "test-1234-5678",
        "faithfulness": 0.85,
        "answer_relevancy": 0.78,
        "context_precision": 0.82,
        "context_recall": 0.91,
        "avg_score": 0.84,
        "testset_size": 5,
        "created_at": "2026-05-30T00:00:00Z",
    })
    runner.doc_store = MagicMock()
    runner.vector_store = MagicMock()
    return runner


def _make_db_row(run_id, faith=0.9, relev=0.8, prec=0.85, recall=0.95,
                 snapshot=None):
    """Helper: 构造匹配实际 PG 表结构的 mock 行."""
    import json
    if snapshot is None:
        snapshot = {"avg_score": 0.875, "testset_size": 10, "details": []}
    return {
        "id": f"db-{run_id}",
        "run_id": run_id,
        "config_label": "ragas_eval",
        "faithfulness": faith,
        "answer_relevancy": relev,
        "context_precision": prec,
        "context_recall": recall,
        "config_snapshot": json.dumps(snapshot) if isinstance(snapshot, dict) else snapshot,
        "created_at": "2026-05-30T00:00:00Z",
    }


@pytest.fixture
def test_app(mock_eval_runner):
    """创建带 mock evaluation_runner 的测试应用."""
    from fastapi import FastAPI
    from src.api.evaluation_routes import evaluation_router

    app = FastAPI()
    app.include_router(evaluation_router)
    app.state.evaluation_runner = mock_eval_runner
    app.state.graph = None

    return app


@pytest.fixture
def client(test_app):
    """TestClient."""
    return TestClient(test_app)


class TestEvalRunEndpoint:
    """POST /eval/run 测试."""

    def test_auto_mode_returns_200(self, client, mock_eval_runner):
        response = client.post(
            "/eval/run",
            json={
                "testset_source": "auto",
                "document_ids": ["doc-1"],
                "testset_size": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["evaluation_id"] == "test-1234-5678"

    def test_manual_mode_returns_200(self, client, mock_eval_runner):
        response = client.post(
            "/eval/run",
            json={
                "testset_source": "manual",
                "testset": [
                    {"question": "什么是RAG", "ground_truth": "RAG是检索增强生成"}
                ],
            },
        )
        assert response.status_code == 200

    def test_invalid_testset_source_returns_422(self, client):
        response = client.post("/eval/run", json={"testset_source": "invalid"})
        assert response.status_code == 422

    def test_eval_runner_not_initialized_returns_503(self):
        from fastapi import FastAPI
        from src.api.evaluation_routes import evaluation_router

        app_no = FastAPI()
        app_no.include_router(evaluation_router)
        client = TestClient(app_no)
        response = client.post("/eval/run", json={"testset_source": "auto"})
        assert response.status_code == 503


class TestEvalResultsEndpoint:
    """GET /eval/results 测试."""

    def test_returns_empty_results(self, client, mock_eval_runner):
        mock_eval_runner.doc_store.fetchval = AsyncMock(return_value=0)
        mock_eval_runner.doc_store.fetch = AsyncMock(return_value=[])

        response = client.get("/eval/results")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["results"] == []

    def test_returns_results_paginated(self, client, mock_eval_runner):
        mock_eval_runner.doc_store.fetchval = AsyncMock(return_value=5)
        mock_eval_runner.doc_store.fetch = AsyncMock(
            return_value=[_make_db_row("eval-1")]
        )

        response = client.get("/eval/results?limit=5&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["results"]) == 1
        assert data["results"][0]["evaluation_id"] == "eval-1"

    def test_eval_runner_not_initialized_returns_503(self):
        from fastapi import FastAPI
        from src.api.evaluation_routes import evaluation_router

        app = FastAPI()
        app.include_router(evaluation_router)
        client = TestClient(app)
        response = client.get("/eval/results")
        assert response.status_code == 503


class TestEvalDashboardEndpoint:
    """GET /eval/dashboard 测试."""

    def test_returns_html_dashboard(self, client, mock_eval_runner):
        mock_eval_runner.doc_store.fetchval = AsyncMock(return_value=1)
        mock_eval_runner.doc_store.fetch = AsyncMock(
            return_value=[_make_db_row("eval-1")]
        )

        response = client.get("/eval/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_dashboard_empty_history_returns_html(self, client, mock_eval_runner):
        mock_eval_runner.doc_store.fetchval = AsyncMock(return_value=0)
        mock_eval_runner.doc_store.fetch = AsyncMock(return_value=[])

        response = client.get("/eval/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
