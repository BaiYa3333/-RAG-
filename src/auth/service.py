"""API key management — generation, hashing, validation, revocation, and user CRUD."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone

from src.config import settings
from src.stores.doc_store import DocStore
from src.utils.logger import logger


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key.

    Returns (raw_key, key_hash) — the raw key is shown only once.
    """
    raw = "rag_" + secrets.token_urlsafe(32)
    return raw, _hash_key(raw)


class AuthService:
    """Stateless service wrapping DocStore for auth operations."""

    def __init__(self, doc_store: DocStore):
        self._db = doc_store

    # ── API Key management ────────────────────────────────────

    async def create_api_key(
        self, user_id: str, role: str = "user", label: str | None = None
    ) -> tuple[str, str]:
        """Create a new API key. Returns (raw_key, key_id)."""
        raw_key, key_hash = generate_api_key()
        row = await self._db.fetchrow(
            """INSERT INTO api_keys (user_id, role, label, key_hash)
               VALUES ($1, $2, $3, $4)
               RETURNING id""",
            user_id, role, label, key_hash,
        )
        key_id = str(row["id"])
        logger.info("api_key_created", key_id=key_id, user_id=user_id, role=role)
        return raw_key, key_id

    async def validate_api_key(self, raw_key: str) -> dict | None:
        """Validate a raw API key. Returns user dict or None."""
        key_hash = _hash_key(raw_key)
        row = await self._db.fetchrow(
            """SELECT id, user_id, role, revoked, last_used
               FROM api_keys
               WHERE key_hash = $1""",
            key_hash,
        )
        if not row:
            return None
        if row["revoked"]:
            logger.info("api_key_revoked_used", key_id=str(row["id"]))
            return None

        # Update last_used
        try:
            await self._db.execute(
                "UPDATE api_keys SET last_used = $1 WHERE id = $2",
                datetime.now(timezone.utc), row["id"],
            )
        except Exception:
            pass  # best-effort

        return {"user_id": row["user_id"], "role": row["role"], "api_key_id": str(row["id"])}

    async def list_api_keys(self) -> list[dict]:
        """List all API keys with masked key values."""
        rows = await self._db.fetch(
            """SELECT id, user_id, role, label, revoked, created_at, last_used
               FROM api_keys ORDER BY created_at DESC"""
        )
        return [
            {
                "id": str(r["id"]),
                "user_id": r["user_id"],
                "role": r["role"],
                "label": r.get("label"),
                "revoked": r["revoked"],
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "last_used": r["last_used"].isoformat() if r.get("last_used") else None,
            }
            for r in rows
        ]

    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key. Returns True if found and revoked."""
        result = await self._db.execute(
            "UPDATE api_keys SET revoked = true WHERE id = $1::uuid",
            key_id,
        )
        affected = "UPDATE 1" in result if result else False
        if affected:
            logger.info("api_key_revoked", key_id=key_id)
        return affected

    # ── Usage logging ─────────────────────────────────────────

    async def log_usage(
        self,
        user_id: str | None,
        endpoint: str,
        tokens_used: int = 0,
        latency_ms: float = 0,
        model: str | None = None,
        estimated_cost: float = 0,
        metadata: dict | None = None,
    ) -> None:
        """Record a usage log entry."""
        try:
            await self._db.execute(
                """INSERT INTO usage_logs (user_id, endpoint, model, tokens_used, latency_ms, estimated_cost, metadata)
                   VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)""",
                user_id, endpoint, model, tokens_used, latency_ms, estimated_cost,
                metadata or {},
            )
        except Exception as e:
            logger.warning("usage_log_failed", error=str(e))

    async def get_usage(
        self, user_id: str | None = None, from_date: str | None = None, to_date: str | None = None
    ) -> dict:
        """Aggregate usage stats."""
        conditions = []
        params: list = []
        idx = 1

        if user_id:
            conditions.append(f"user_id = ${idx}")
            params.append(user_id)
            idx += 1
        if from_date:
            conditions.append(f"created_at >= ${idx}::timestamptz")
            params.append(from_date)
            idx += 1
        if to_date:
            conditions.append(f"created_at <= ${idx}::timestamptz")
            params.append(to_date)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        row = await self._db.fetchrow(
            f"""SELECT
                  COUNT(*)::int AS total_requests,
                  COALESCE(SUM(tokens_used), 0)::int AS total_tokens,
                  COALESCE(AVG(latency_ms), 0)::float AS avg_latency_ms,
                  COALESCE(SUM(estimated_cost), 0)::float AS total_cost
               FROM usage_logs {where}""",
            *params,
        )
        return dict(row) if row else {"total_requests": 0, "total_tokens": 0, "avg_latency_ms": 0, "total_cost": 0}
