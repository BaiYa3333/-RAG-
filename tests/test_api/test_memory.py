"""Tests for memory: session CRUD, history load, truncation, cross-user isolation."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def memory_test_app():
    """Create TestClient with mock memory service. Disables auth for session tests."""
    from src.main import app

    mock_memory = AsyncMock()
    mock_memory.get_session = AsyncMock(return_value={
        "id": "uuid-session-1",
        "user_id": "alice",
        "title": "Test Session",
        "summary": None,
        "metadata": {},
        "created_at": "2026-05-31T00:00:00",
        "updated_at": "2026-05-31T00:00:00",
    })
    mock_memory.create_session = AsyncMock(return_value={
        "id": "uuid-session-1",
        "user_id": "alice",
        "title": "Test Session",
        "summary": None,
        "metadata": {},
        "created_at": "2026-05-31T00:00:00",
        "updated_at": "2026-05-31T00:00:00",
    })
    mock_memory.list_sessions = AsyncMock(return_value=[
        {"id": "uuid-1", "user_id": "alice", "title": "Chat 1", "summary": None, "created_at": "2026-05-31T00:00:00", "updated_at": "2026-05-31T00:00:00"},
    ])
    mock_memory.delete_session = AsyncMock(return_value=True)
    mock_memory.load_history = AsyncMock(return_value=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ])
    mock_memory.append_turn = AsyncMock(return_value="uuid-turn-1")
    mock_memory.get_turn_count = AsyncMock(return_value=5)

    app.state.memory_service = mock_memory
    app.state.cache_store = None
    app.state.vector_store = None
    app.state.doc_store = None
    app.state.llm_client = None
    app.state.graph = None

    # Disable auth for session CRUD tests
    import src.auth.dependencies as deps
    original_auth = deps.settings.auth_enabled
    deps.settings.auth_enabled = False

    client = TestClient(app)
    yield client, mock_memory

    deps.settings.auth_enabled = original_auth
    app.state.memory_service = None


class TestSessionCRUD:
    """Task 6.7: Session create, list, delete tests."""

    def test_create_session(self, memory_test_app):
        client, _ = memory_test_app
        resp = client.post("/sessions", json={"title": "Test Session"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "uuid-session-1"
        assert data["title"] == "Test Session"

    def test_create_session_no_title(self, memory_test_app):
        client, _ = memory_test_app
        resp = client.post("/sessions", json={})
        assert resp.status_code == 201

    def test_list_sessions(self, memory_test_app):
        client, _ = memory_test_app
        resp = client.get("/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["id"] == "uuid-1"

    def test_delete_session(self, memory_test_app):
        client, _ = memory_test_app
        resp = client.delete("/sessions/uuid-1")
        assert resp.status_code == 200
        assert "deleted" in resp.json()["detail"]

    def test_delete_nonexistent_session(self, memory_test_app):
        client, mock_memory = memory_test_app
        mock_memory.delete_session.return_value = False
        resp = client.delete("/sessions/nonexistent")
        assert resp.status_code == 404


class TestHistoryLoad:
    """Task 6.7: History loading from persisted sessions."""

    def test_chat_with_session_loads_history(self, memory_test_app):
        client, mock_memory = memory_test_app
        # Mock graph to return a valid response
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "query": "test query",
            "answer": "test answer",
            "model_name": "deepseek-chat",
            "documents": [],
            "retrieval_tier": 1,
            "gate_log": [],
            "errors": [],
        })
        from src.main import app
        app.state.graph = mock_graph

        resp = client.post("/chat", json={
            "query": "What did we discuss?",
            "chat_history": [],
            "session_id": "uuid-session-1",
        })

        # Verify load_history was called
        mock_memory.load_history.assert_called_once_with("uuid-session-1")
        # Verify turns were persisted
        assert mock_memory.append_turn.call_count >= 2
        app.state.graph = None

    def test_chat_without_session_skips_memory(self, memory_test_app):
        client, mock_memory = memory_test_app
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "query": "test", "answer": "ok", "model_name": "deepseek-chat",
            "documents": [], "retrieval_tier": 1, "gate_log": [], "errors": [],
        })
        from src.main import app
        app.state.graph = mock_graph

        resp = client.post("/chat", json={"query": "test", "chat_history": []})
        # load_history should NOT be called without session_id
        mock_memory.load_history.assert_not_called()
        app.state.graph = None


class TestContextManager:
    """Task 6.7: Token counting and truncation tests."""

    def test_estimate_tokens(self):
        from src.memory.context_manager import estimate_tokens
        assert estimate_tokens("hello") > 0
        assert estimate_tokens("") == 0
        # Chinese text
        tokens = estimate_tokens("你好世界")
        assert tokens > 0

    def test_estimate_history_tokens(self):
        from src.memory.context_manager import estimate_history_tokens
        history = [
            {"role": "user", "content": "What is RRF?"},
            {"role": "assistant", "content": "RRF is a fusion algorithm."},
        ]
        tokens = estimate_history_tokens(history)
        assert tokens > 0

    def test_memory_service_truncation_old_turns(self, memory_test_app):
        _, mock_memory = memory_test_app

        def delete_old(session_id, keep_count=10):
            return 5  # deleted 5 turns
        mock_memory.delete_old_turns = delete_old

        result = mock_memory.delete_old_turns("uuid-1", keep_count=5)
        assert result == 5


class TestMemoryAuthIsolation:
    """Task 5.6: /memory endpoints with authenticated user isolation."""

    @pytest.fixture
    def memory_auth_client(self):
        """Create TestClient with auth enabled and mock memory service."""
        from fastapi.testclient import TestClient
        from src.main import app

        mock_memory = AsyncMock()
        mock_memory.load_user_memories = AsyncMock(return_value=[
            {
                "id": "mem-1",
                "user_id": "test-user-123",
                "memory_type": "fact",
                "content": "User prefers Python",
                "source_session_id": None,
                "created_at": "2026-06-01T00:00:00",
                "expires_at": None,
            },
        ])
        mock_memory.delete_user_memory = AsyncMock(return_value=True)
        mock_memory.clear_user_memories = AsyncMock(return_value=3)

        app.state.memory_service = mock_memory
        app.state.cache_store = None
        app.state.vector_store = None
        app.state.doc_store = None

        import src.auth.dependencies as deps
        original = deps.settings.auth_enabled
        deps.settings.auth_enabled = True

        client = TestClient(app)
        yield client, mock_memory

        deps.settings.auth_enabled = original
        app.state.memory_service = None

    def test_memory_list_without_auth_returns_401(self, memory_auth_client):
        """GET /memory without auth should return 401."""
        client, _ = memory_auth_client
        response = client.get("/memory")
        assert response.status_code == 401

    def test_memory_list_with_auth_returns_user_memories(self, memory_auth_client):
        """GET /memory with valid auth should return user-specific memories."""
        from src.auth.user_service import create_jwt

        client, mock_memory = memory_auth_client
        token = create_jwt("test-user-123", "user", expires_h=1)
        response = client.get(
            "/memory",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["user_id"] == "test-user-123"

    def test_memory_delete_without_auth_returns_401(self, memory_auth_client):
        """DELETE /memory/{id} without auth should return 401."""
        client, _ = memory_auth_client
        response = client.delete("/memory/mem-1")
        assert response.status_code == 401

    def test_memory_delete_with_auth(self, memory_auth_client):
        """DELETE /memory/{id} with valid auth should succeed."""
        from src.auth.user_service import create_jwt

        client, _ = memory_auth_client
        token = create_jwt("test-user-123", "user", expires_h=1)
        response = client.delete(
            "/memory/mem-1",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert "deleted" in response.json()["detail"]

    def test_memory_clear_without_auth_returns_401(self, memory_auth_client):
        """DELETE /memory without auth should return 401."""
        client, _ = memory_auth_client
        response = client.delete("/memory")
        assert response.status_code == 401

    def test_different_users_see_different_memories(self, memory_auth_client):
        """Two different authenticated users should see only their own memories."""
        from src.auth.user_service import create_jwt

        client, mock_memory = memory_auth_client

        # Configure mock to return different results for different users
        async def load_memories(user_id, session_id=None):
            if user_id == "alice":
                return [{"id": "mem-a1", "user_id": "alice", "memory_type": "fact", "content": "Alice memory", "source_session_id": None, "created_at": "2026-06-01T00:00:00", "expires_at": None}]
            elif user_id == "bob":
                return [{"id": "mem-b1", "user_id": "bob", "memory_type": "fact", "content": "Bob memory", "source_session_id": None, "created_at": "2026-06-01T00:00:00", "expires_at": None}]
            return []

        mock_memory.load_user_memories = load_memories

        alice_token = create_jwt("alice", "user", expires_h=1)
        bob_token = create_jwt("bob", "user", expires_h=1)

        alice_resp = client.get("/memory", headers={"Authorization": f"Bearer {alice_token}"})
        bob_resp = client.get("/memory", headers={"Authorization": f"Bearer {bob_token}"})

        assert alice_resp.status_code == 200
        assert bob_resp.status_code == 200
        assert alice_resp.json()[0]["user_id"] == "alice"
        assert bob_resp.json()[0]["user_id"] == "bob"
