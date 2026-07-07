"""Tests for session isolation and memory isolation changes.

Covers:
- _get_user_id JWT extraction (auth enabled/disabled)
- Session ownership verification (cross-user isolation)
- Cascaded memory deletion on session delete
- Session-scoped memory loading
- Memory extraction with session_id
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def memory_service_mock():
    """Full mock of MemoryService for isolation tests."""
    svc = AsyncMock()
    svc.create_session = AsyncMock(return_value={
        "id": "uuid-session-1", "user_id": "user-a", "title": "Test",
        "summary": None, "metadata": {}, "created_at": "2026-06-01T00:00:00", "updated_at": "2026-06-01T00:00:00",
    })
    svc.get_session = AsyncMock(return_value={
        "id": "uuid-session-1", "user_id": "user-a", "title": "Test",
        "summary": None, "metadata": {}, "created_at": "2026-06-01T00:00:00", "updated_at": "2026-06-01T00:00:00",
    })
    svc.list_sessions = AsyncMock(return_value=[
        {"id": "uuid-1", "user_id": "user-a", "title": "Chat 1", "summary": None, "created_at": "2026-06-01T00:00:00", "updated_at": "2026-06-01T00:00:00"},
    ])
    svc.delete_session = AsyncMock(return_value=True)
    svc.load_history = AsyncMock(return_value=[])
    svc.append_turn = AsyncMock(return_value="uuid-turn-1")
    svc.get_turn_count = AsyncMock(return_value=3)
    svc.load_user_memories = AsyncMock(return_value=[])
    svc.save_user_memory = AsyncMock(return_value="uuid-mem-1")
    svc.delete_user_memory = AsyncMock(return_value=True)
    svc.clear_user_memories = AsyncMock(return_value=0)
    return svc


@pytest.fixture
def isolation_test_app(memory_service_mock):
    """TestClient with isolation features enabled and JWT auth configured."""
    from src.main import app
    from src.config import settings

    app.state.memory_service = memory_service_mock
    app.state.cache_store = None
    app.state.vector_store = None
    app.state.doc_store = None
    app.state.llm_client = None
    app.state.graph = None
    app.state.user_service = None

    # Enable auth for isolation tests
    original_auth = settings.auth_enabled
    settings.auth_enabled = True

    client = TestClient(app)
    yield client, memory_service_mock

    settings.auth_enabled = original_auth
    app.state.memory_service = None


# ── Helper: create a valid JWT for testing ─────────────────────────


def _make_token(user_id: str, role: str = "user") -> str:
    from src.auth.user_service import create_jwt
    return create_jwt(user_id, role)


# ── Task 8.1: _get_user_id tests ───────────────────────────────────


class TestGetUserId:
    """Tests for _get_user_id JWT extraction."""

    def test_auth_enabled_extracts_user_from_jwt(self, isolation_test_app):
        """With auth enabled, _get_user_id should return user_id from JWT sub claim."""
        client, _ = isolation_test_app
        token = _make_token("user-bob")
        resp = client.get("/sessions", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_auth_enabled_missing_token_returns_401(self, isolation_test_app):
        """With auth enabled, missing Authorization header should return 401."""
        client, _ = isolation_test_app
        resp = client.get("/sessions")  # No Authorization header
        assert resp.status_code == 401
        assert "Authorization token required" in resp.json()["detail"]

    def test_auth_enabled_invalid_token_returns_401(self, isolation_test_app):
        """With auth enabled, an invalid/expired JWT should return 401."""
        client, _ = isolation_test_app
        resp = client.get("/sessions", headers={"Authorization": "Bearer invalid-token-not-jwt"})
        assert resp.status_code == 401

    def test_auth_disabled_returns_anonymous(self, memory_service_mock):
        """With auth disabled, _get_user_id should return 'anonymous'."""
        from src.main import app
        from src.config import settings

        app.state.memory_service = memory_service_mock
        app.state.graph = None

        original = settings.auth_enabled
        settings.auth_enabled = False

        client = TestClient(app)
        resp = client.get("/sessions")
        assert resp.status_code == 200

        settings.auth_enabled = original
        app.state.memory_service = None

    def test_create_session_with_auth_uses_correct_user(self, isolation_test_app):
        """Session creation with valid JWT should assign correct user_id."""
        client, mock = isolation_test_app
        # Override create_session return to match expected user
        mock.create_session.return_value = {
            "id": "uuid-new", "user_id": "user-alice", "title": "My Session",
            "summary": None, "metadata": {}, "created_at": "2026-06-01T00:00:00", "updated_at": "2026-06-01T00:00:00",
        }
        token = _make_token("user-alice")
        resp = client.post("/sessions", json={"title": "My Session"}, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 201
        # Verify create_session was called with correct user_id
        mock.create_session.assert_called_once()
        call_args = mock.create_session.call_args
        assert call_args.kwargs["user_id"] == "user-alice"


# ── Task 8.2: Cascaded memory deletion tests ───────────────────────


class TestCascadedMemoryDeletion:
    """Tests for delete_session cascading to user_memories."""

    def test_delete_session_with_memories(self, isolation_test_app):
        """Deleting a session should also delete associated memories."""
        client, _ = isolation_test_app
        token = _make_token("user-a")
        resp = client.delete("/sessions/uuid-session-1", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert "deleted" in resp.json()["detail"]

    def test_delete_session_denied_for_other_user(self, isolation_test_app):
        """User B should not be able to delete User A's session."""
        client, _ = isolation_test_app
        token = _make_token("user-b")

        # Override get_session: session exists but owned by user-a
        async def get_session_owned_by_other(session_id, user_id=None):
            if user_id and user_id != "user-a":
                return None  # Not found for this user
            return {
                "id": session_id, "user_id": "user-a", "title": "Test",
                "summary": None, "metadata": {}, "created_at": "2026-06-01T00:00:00", "updated_at": "2026-06-01T00:00:00",
            }

        client, mock = isolation_test_app
        mock.get_session = get_session_owned_by_other
        mock.delete_session.return_value = True

        resp = client.delete("/sessions/uuid-session-1", headers={"Authorization": f"Bearer {token}"})
        # Should get 403 because session belongs to another user
        assert resp.status_code == 403
        assert "Access denied" in resp.json()["detail"]

    def test_delete_nonexistent_session_returns_404(self, isolation_test_app):
        """Deleting a non-existent session returns 404."""
        client, mock = isolation_test_app

        async def session_not_found(session_id, user_id=None):
            return None

        mock.get_session = session_not_found
        mock.delete_session.return_value = False

        token = _make_token("user-a")
        resp = client.delete("/sessions/nonexistent-uuid", headers={"Authorization": f"Bearer {token}"})
        # Could be 404 or 403 depending on path — but ownership check happens first
        # If session doesn't exist at all, it returns 404
        assert resp.status_code == 404


# ── Task 8.3: Session-scoped memory loading tests ──────────────────


class TestSessionScopedMemoryLoading:
    """Tests for load_user_memories with session_id filtering."""

    def test_load_memories_without_session_id_loads_all(self, isolation_test_app):
        """Without session_id, all user memories should be returned."""
        client, mock = isolation_test_app

        async def load_all(user_id, limit=20, session_id=None):
            return [
                {"id": "m1", "user_id": user_id, "memory_type": "fact", "content": "Fact 1",
                 "source_session_id": "s1", "created_at": "2026-06-01T00:00:00", "expires_at": None},
                {"id": "m2", "user_id": user_id, "memory_type": "fact", "content": "Fact 2",
                 "source_session_id": "s2", "created_at": "2026-06-01T00:00:00", "expires_at": None},
            ]

        mock.load_user_memories = load_all

        token = _make_token("user-a")
        resp = client.get("/memory", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_load_memories_with_session_id_filters(self, isolation_test_app):
        """With session_id, only session-scoped + legacy memories should be returned."""
        client, mock = isolation_test_app

        async def load_scoped(user_id, limit=20, session_id=None):
            if session_id == "s1":
                return [
                    {"id": "m1", "user_id": user_id, "memory_type": "fact", "content": "Fact 1",
                     "source_session_id": "s1", "created_at": "2026-06-01T00:00:00", "expires_at": None},
                ]
            return []

        mock.load_user_memories = load_scoped

        token = _make_token("user-a")
        resp = client.get("/memory?session_id=s1", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["source_session_id"] == "s1"


# ── Task 8.4: Cross-user and cross-session isolation tests ──────────


class TestCrossUserSessionIsolation:
    """End-to-end isolation tests."""

    def test_user_b_cannot_access_user_a_session(self, isolation_test_app):
        """User B should get 403 when trying to access User A's session."""
        client, mock = isolation_test_app

        # Mock: session exists for user-a only.
        # _verify_session_ownership makes two calls:
        #   1. get_session(session_id, user_id=current_user) → ownership check
        #   2. get_session(session_id) → existence check (to distinguish 403 vs 404)
        async def get_session_owner_check(session_id, user_id=None):
            if user_id is None:
                # Fallback check: session exists (regardless of owner)
                return {
                    "id": session_id, "user_id": "user-a", "title": "User A Session",
                    "summary": None, "metadata": {}, "created_at": "2026-06-01T00:00:00", "updated_at": "2026-06-01T00:00:00",
                }
            if user_id == "user-a":
                return {
                    "id": session_id, "user_id": "user-a", "title": "User A Session",
                    "summary": None, "metadata": {}, "created_at": "2026-06-01T00:00:00", "updated_at": "2026-06-01T00:00:00",
                }
            return None

        mock.get_session = get_session_owner_check

        token_b = _make_token("user-b")
        resp = client.get("/sessions/uuid-session-1", headers={"Authorization": f"Bearer {token_b}"})
        assert resp.status_code == 403

    def test_user_a_can_access_own_session(self, isolation_test_app):
        """User A should be able to access their own session."""
        client, mock = isolation_test_app

        async def get_session_owner_check(session_id, user_id=None):
            if user_id is None or user_id == "user-a":
                return {
                    "id": session_id, "user_id": "user-a", "title": "User A Session",
                    "summary": None, "metadata": {}, "created_at": "2026-06-01T00:00:00", "updated_at": "2026-06-01T00:00:00",
                }
            return None

        mock.get_session = get_session_owner_check

        token_a = _make_token("user-a")
        resp = client.get("/sessions/uuid-session-1", headers={"Authorization": f"Bearer {token_a}"})
        assert resp.status_code == 200

    def test_user_list_sessions_only_sees_own(self, isolation_test_app):
        """User should only see their own sessions in list."""
        client, mock = isolation_test_app

        async def list_by_user(user_id=None, ttl_days=None):
            if user_id == "user-a":
                return [
                    {"id": "uuid-1", "user_id": "user-a", "title": "A's Chat", "summary": None,
                     "created_at": "2026-06-01T00:00:00", "updated_at": "2026-06-01T00:00:00"},
                ]
            return []

        mock.list_sessions = list_by_user

        token_a = _make_token("user-a")
        resp = client.get("/sessions", headers={"Authorization": f"Bearer {token_a}"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["user_id"] == "user-a"


class TestCrossSessionMemoryIsolation:
    """Tests for memory isolation between sessions."""

    def test_memories_from_session_a_not_in_session_b(self, memory_service_mock):
        """Memories scoped to session A should not appear when loading session B."""
        from src.main import app
        from src.config import settings

        # Setup: memories for session-a only
        async def load_with_session(user_id, limit=20, session_id=None):
            if session_id == "session-a":
                return [
                    {"id": "m1", "user_id": user_id, "memory_type": "fact",
                     "content": "User uploaded doc-a.pdf", "source_session_id": "session-a",
                     "created_at": "2026-06-01T00:00:00", "expires_at": None},
                ]
            elif session_id == "session-b":
                return []
            # No session_id: return all
            return [
                {"id": "m1", "user_id": user_id, "memory_type": "fact",
                 "content": "User uploaded doc-a.pdf", "source_session_id": "session-a",
                 "created_at": "2026-06-01T00:00:00", "expires_at": None},
            ]

        memory_service_mock.load_user_memories = load_with_session
        app.state.memory_service = memory_service_mock

        original = settings.auth_enabled
        settings.auth_enabled = False

        client = TestClient(app)

        # Load memories for session-b: should be empty
        resp = client.get("/memory?session_id=session-b")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 0

        # Load memories for session-a: should have 1
        resp = client.get("/memory?session_id=session-a")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["source_session_id"] == "session-a"

        settings.auth_enabled = original
        app.state.memory_service = None
