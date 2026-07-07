"""Unit tests for validate_auth_secrets in config.py (Task 5.1)."""

import os
from unittest.mock import patch

import pytest


class TestValidateAuthSecrets:
    """Task 5.1: validate_auth_secrets tests."""

    def test_production_weak_jwt_secret_rejected(self, monkeypatch):
        """Production with weak JWT secret should raise ValueError."""
        # Import BEFORE monkeypatch so the module-level `settings = Settings()`
        # succeeds with dev defaults (avoids import-time ValidationError).
        from src.config import Settings

        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("RAG_JWT_SECRET", "change-me-in-production-use-random-secret")
        monkeypatch.setenv("RAG_ADMIN_SECRET_CODE", "strong-code-12345")

        with pytest.raises(ValueError, match="RAG_JWT_SECRET must be set to a strong"):
            Settings()

    def test_production_weak_admin_code_no_longer_rejected(self, monkeypatch):
        """Production with weak admin code should no longer raise — admin code is deprecated."""
        from src.config import Settings

        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("RAG_JWT_SECRET", "strong-jwt-secret-12345")
        monkeypatch.setenv("RAG_ADMIN_SECRET_CODE", "321458")

        # Admin code validation removed — should succeed
        settings = Settings()
        assert settings.admin_secret_code == "321458"

    def test_development_weak_secrets_allowed(self, monkeypatch):
        """Development with weak secrets should NOT raise ValueError."""
        monkeypatch.setenv("ENV", "development")
        monkeypatch.setenv("RAG_JWT_SECRET", "change-me-in-production-use-random-secret")
        monkeypatch.setenv("RAG_ADMIN_SECRET_CODE", "321458")

        from src.config import Settings
        # Should not raise
        settings = Settings()
        assert settings.jwt_secret == "change-me-in-production-use-random-secret"

    def test_development_env_unset_allows_weak_secrets(self, monkeypatch):
        """Unset ENV (defaults to development) with weak secrets should not raise."""
        monkeypatch.delenv("ENV", raising=False)
        monkeypatch.setenv("RAG_JWT_SECRET", "change-me-in-production-use-random-secret")
        monkeypatch.setenv("RAG_ADMIN_SECRET_CODE", "321458")

        from src.config import Settings
        # Should not raise
        settings = Settings()
        assert settings.jwt_secret == "change-me-in-production-use-random-secret"

    def test_production_strong_secrets_pass(self, monkeypatch):
        """Production with strong secrets should pass validation."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("RAG_JWT_SECRET", "my-strong-jwt-secret-abc123")
        monkeypatch.setenv("RAG_ADMIN_SECRET_CODE", "my-strong-admin-code-xyz789")

        from src.config import Settings
        settings = Settings()
        assert settings.jwt_secret == "my-strong-jwt-secret-abc123"
        assert settings.admin_secret_code == "my-strong-admin-code-xyz789"

    def test_unsafe_constants_are_frozensets(self):
        """UNSAFE_JWT_SECRETS and UNSAFE_ADMIN_CODES should be frozenset constants."""
        from src.config import UNSAFE_JWT_SECRETS, UNSAFE_ADMIN_CODES

        assert isinstance(UNSAFE_JWT_SECRETS, frozenset)
        assert isinstance(UNSAFE_ADMIN_CODES, frozenset)
        assert "change-me-in-production-use-random-secret" in UNSAFE_JWT_SECRETS
        assert "321458" in UNSAFE_ADMIN_CODES

    def test_unsafe_constants_are_immutable(self):
        """frozenset constants should not support mutation."""
        from src.config import UNSAFE_JWT_SECRETS

        # frozenset has no mutating methods — attempting mutation raises AttributeError
        with pytest.raises((TypeError, AttributeError)):
            UNSAFE_JWT_SECRETS.add("new-secret")  # type: ignore
