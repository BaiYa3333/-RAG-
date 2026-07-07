"""Tests for auth: API key validation, role-based access, key revocation."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def auth_test_app():
    """Create TestClient with auth enabled and mock auth service."""
    from src.main import app

    # Mock doc_store and auth_service
    mock_auth = AsyncMock()
    mock_auth.validate_api_key = AsyncMock(return_value=None)  # default: invalid
    mock_auth.create_api_key = AsyncMock(return_value=("raw-key-123", "uuid-1"))
    mock_auth.list_api_keys = AsyncMock(return_value=[
        {"id": "uuid-1", "user_id": "alice", "role": "admin", "label": "prod", "revoked": False, "created_at": None, "last_used": None},
    ])
    mock_auth.revoke_api_key = AsyncMock(return_value=True)
    mock_auth.get_usage = AsyncMock(return_value={"total_requests": 10, "total_tokens": 500, "avg_latency_ms": 150.0, "total_cost": 0.01})

    app.state.auth_service = mock_auth
    app.state.cache_store = None
    app.state.vector_store = None
    app.state.doc_store = None
    app.state.llm_client = None
    app.state.graph = None

    # Temporarily enable auth
    import src.auth.dependencies as deps
    original = deps.settings.auth_enabled
    deps.settings.auth_enabled = True

    client = TestClient(app)
    yield client, mock_auth

    # Restore
    deps.settings.auth_enabled = original
    app.state.auth_service = None


class TestApiKeyValidation:
    """Task 4.7: API key validation tests."""

    def test_valid_key_proceeds(self, auth_test_app):
        client, mock_auth = auth_test_app
        mock_auth.validate_api_key.return_value = {"user_id": "alice", "role": "admin", "api_key_id": "uuid-1"}

        # Test that admin endpoints accept valid key
        resp = client.get("/admin/api-keys", headers={"Authorization": "Bearer sk-valid"})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_invalid_key_returns_401(self, auth_test_app):
        client, mock_auth = auth_test_app
        mock_auth.validate_api_key.return_value = None  # invalid

        resp = client.get("/admin/api-keys", headers={"Authorization": "Bearer sk-invalid"})
        assert resp.status_code == 401
        assert "Invalid credentials" in resp.json()["detail"]

    def test_missing_key_when_auth_required(self, auth_test_app):
        client, _ = auth_test_app

        resp = client.get("/admin/api-keys")  # No Authorization header
        assert resp.status_code == 401
        assert "Authorization token required" in resp.json()["detail"]

    def test_auth_disabled_allows_access(self, auth_test_app):
        client, mock_auth = auth_test_app
        # Disable auth
        import src.auth.dependencies as deps
        deps.settings.auth_enabled = False

        mock_auth.validate_api_key.return_value = {"user_id": "anonymous", "role": "user", "api_key_id": None}

        # Chat endpoint should work without auth
        resp = client.post("/chat", json={"query": "test", "chat_history": []})
        # This will hit the graph which is None in test, but auth should pass
        assert resp.status_code in (200, 503)  # 503 = graph not available (OK, auth passed)


class TestRoleBasedAccess:
    """Task 4.7: Role-based access control tests."""

    def test_admin_access_allowed(self, auth_test_app):
        client, mock_auth = auth_test_app
        mock_auth.validate_api_key.return_value = {"user_id": "admin1", "role": "admin", "api_key_id": "uuid-1"}

        resp = client.post("/admin/api-keys", json={"user_id": "bob", "role": "user"}, headers={"Authorization": "Bearer sk-admin"})
        assert resp.status_code == 200
        data = resp.json()
        assert "raw_key" in data

    def test_any_authenticated_user_can_access_admin_api(self, auth_test_app):
        client, mock_auth = auth_test_app
        mock_auth.validate_api_key.return_value = {"user_id": "bob", "role": "user", "api_key_id": "uuid-2"}

        resp = client.post("/admin/api-keys", json={"user_id": "charlie"}, headers={"Authorization": "Bearer sk-user"})
        assert resp.status_code == 200
        assert "raw_key" in resp.json()

    def test_any_authenticated_user_can_access_admin_api_viewer(self, auth_test_app):
        client, mock_auth = auth_test_app
        mock_auth.validate_api_key.return_value = {"user_id": "viewer1", "role": "viewer", "api_key_id": "uuid-3"}

        resp = client.post("/admin/api-keys", json={"user_id": "dave"}, headers={"Authorization": "Bearer sk-viewer"})
        assert resp.status_code == 200
        assert "raw_key" in resp.json()


class TestKeyRevocation:
    """Task 4.7: Key revocation tests."""

    def test_revoke_active_key(self, auth_test_app):
        client, mock_auth = auth_test_app
        mock_auth.validate_api_key.return_value = {"user_id": "admin1", "role": "admin", "api_key_id": "uuid-1"}
        mock_auth.revoke_api_key.return_value = True

        resp = client.delete("/admin/api-keys/uuid-2", headers={"Authorization": "Bearer sk-admin"})
        assert resp.status_code == 200
        assert "revoked" in resp.json()["detail"]

    def test_revoke_nonexistent_key(self, auth_test_app):
        client, mock_auth = auth_test_app
        mock_auth.validate_api_key.return_value = {"user_id": "admin1", "role": "admin", "api_key_id": "uuid-1"}
        mock_auth.revoke_api_key.return_value = False

        resp = client.delete("/admin/api-keys/nonexistent", headers={"Authorization": "Bearer sk-admin"})
        assert resp.status_code == 404

    def test_revoked_key_cannot_access(self, auth_test_app):
        client, mock_auth = auth_test_app
        mock_auth.validate_api_key.return_value = None  # Simulate revoked key

        resp = client.get("/admin/usage", headers={"Authorization": "Bearer sk-revoked"})
        assert resp.status_code == 401


class TestUsageEndpoint:
    """Task 4.7: Usage analytics tests."""

    def test_usage_with_filters(self, auth_test_app):
        client, mock_auth = auth_test_app
        mock_auth.validate_api_key.return_value = {"user_id": "admin1", "role": "admin", "api_key_id": "uuid-1"}

        resp = client.get(
            "/admin/usage?user_id=alice&from_date=2026-05-01&to_date=2026-05-31",
            headers={"Authorization": "Bearer sk-admin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "total_tokens" in data

    def test_usage_accessible_by_any_authenticated_user(self, auth_test_app):
        client, mock_auth = auth_test_app
        mock_auth.validate_api_key.return_value = {"user_id": "bob", "role": "user", "api_key_id": "uuid-2"}

        resp = client.get("/admin/usage", headers={"Authorization": "Bearer sk-user"})
        assert resp.status_code == 200
        assert "total_requests" in resp.json()
