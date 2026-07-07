"""Session service — CRUD for sessions and conversations using DocStore (asyncpg)."""

from __future__ import annotations

from datetime import datetime, timezone

from src.config import settings
from src.stores.doc_store import DocStore
from src.utils.logger import logger


class MemoryService:
    """Stateless service wrapping DocStore for session and conversation operations."""

    def __init__(self, doc_store: DocStore):
        self._db = doc_store

    # ── Session CRUD ───────────────────────────────────────────

    async def create_session(self, user_id: str | None = None, title: str | None = None) -> dict:
        """Create a new conversation session."""
        row = await self._db.fetchrow(
            """INSERT INTO sessions (user_id, title)
               VALUES ($1, $2)
               RETURNING id, user_id, title, summary, metadata, created_at, updated_at""",
            user_id, title,
        )
        logger.info("session_created", session_id=str(row["id"]))
        return {
            "id": str(row["id"]),
            "user_id": row["user_id"],
            "title": row.get("title"),
            "summary": row.get("summary"),
            "metadata": row.get("metadata") or {},
            "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
            "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        }

    async def list_sessions(self, user_id: str | None = None, ttl_days: int | None = None) -> list[dict]:
        """List sessions, optionally filtered by user. Ordered by last activity.

        When ttl_days is set, sessions with updated_at older than ttl_days are excluded.
        """
        from datetime import datetime, timedelta, timezone

        conditions = []
        params: list = []
        idx = 1

        if user_id:
            conditions.append(f"user_id = ${idx}")
            params.append(user_id)
            idx += 1

        if ttl_days and ttl_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
            conditions.append(f"updated_at > ${idx}")
            params.append(cutoff)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._db.fetch(
            f"""SELECT id, user_id, title, summary, metadata, created_at, updated_at
               FROM sessions
               {where}
               ORDER BY updated_at DESC""",
            *params,
        )
        return [
            {
                "id": str(r["id"]),
                "user_id": r["user_id"],
                "title": r.get("title"),
                "summary": r.get("summary"),
                "metadata": r.get("metadata") or {},
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
            }
            for r in rows
        ]

    async def get_session(self, session_id: str, user_id: str | None = None) -> dict | None:
        """Get a single session by ID. Optionally filter by user_id for ownership check."""
        if user_id:
            row = await self._db.fetchrow(
                """SELECT id, user_id, title, summary, metadata, created_at, updated_at
                   FROM sessions WHERE id = $1::uuid AND user_id = $2""",
                session_id, user_id,
            )
        else:
            row = await self._db.fetchrow(
                """SELECT id, user_id, title, summary, metadata, created_at, updated_at
                   FROM sessions WHERE id = $1::uuid""",
                session_id,
            )
        if not row:
            return None
        return {
            "id": str(row["id"]),
            "user_id": row["user_id"],
            "title": row.get("title"),
            "summary": row.get("summary"),
            "metadata": row.get("metadata") or {},
            "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
            "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        }

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session, its conversations, and associated user memories.

        Conversations are deleted via FK CASCADE.
        User memories are deleted explicitly (dual safeguard: app layer + FK CASCADE).
        Session-scoped ChromaDB collection is cleaned up (best-effort).
        """
        # Delete associated user memories first (also handled by FK CASCADE as safeguard)
        mem_result = await self._db.execute(
            "DELETE FROM user_memories WHERE source_session_id = $1::uuid", session_id
        )
        mem_count = 0
        if mem_result:
            try:
                mem_count = int(mem_result.split()[-1]) if "DELETE" in mem_result else 0
            except (ValueError, IndexError):
                mem_count = 0
        if mem_count > 0:
            logger.info("session_memories_deleted", session_id=session_id, count=mem_count)

        result = await self._db.execute(
            "DELETE FROM sessions WHERE id = $1::uuid", session_id
        )
        affected = "DELETE 1" in result if result else False
        if affected:
            logger.info("session_deleted", session_id=session_id)

        # Best-effort cleanup of session-scoped ChromaDB collection
        if affected:
            try:
                from src.rag.retrieval.dense import _cleanup_session_collection
                await _cleanup_session_collection(session_id)
            except Exception as e:
                logger.warning(
                    "session_vector_cleanup_failed",
                    session_id=session_id,
                    error=str(e),
                )

        return affected

    # ── Conversation CRUD ──────────────────────────────────────

    async def append_turn(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> str:
        """Append a conversation turn. Returns the turn ID."""
        import json as _json
        meta_json = _json.dumps(metadata or {}, ensure_ascii=False)
        row = await self._db.fetchrow(
            """INSERT INTO conversations (session_id, role, content, metadata)
               VALUES ($1::uuid, $2, $3, $4::jsonb)
               RETURNING id""",
            session_id, role, content, meta_json,
        )
        # Bump session updated_at
        await self._db.execute(
            "UPDATE sessions SET updated_at = $1 WHERE id = $2::uuid",
            datetime.now(timezone.utc), session_id,
        )
        return str(row["id"])

    async def load_history(
        self, session_id: str, max_turns: int | None = None
    ) -> list[dict]:
        """Load conversation history for a session, most recent first.

        Returns list of {role, content} dicts in chronological order.
        """
        limit_clause = f"LIMIT {max_turns}" if max_turns else ""
        rows = await self._db.fetch(
            f"""SELECT role, content, metadata, created_at
               FROM conversations
               WHERE session_id = $1::uuid
               ORDER BY created_at ASC
               {limit_clause}""",
            session_id,
        )
        # Load summary if present
        session = await self.get_session(session_id)
        history: list[dict] = []
        if session and session.get("summary"):
            history.append({"role": "system", "content": f"[Previous conversation summary]: {session['summary']}"})

        for r in rows:
            history.append({"role": r["role"], "content": r["content"]})
        return history

    async def get_turn_count(self, session_id: str) -> int:
        """Get the number of conversation turns in a session."""
        row = await self._db.fetchrow(
            "SELECT COUNT(*)::int AS cnt FROM conversations WHERE session_id = $1::uuid",
            session_id,
        )
        return row["cnt"] if row else 0

    async def update_summary(self, session_id: str, summary: str) -> None:
        """Update the session summary."""
        await self._db.execute(
            "UPDATE sessions SET summary = $1, updated_at = $2 WHERE id = $3::uuid",
            summary, datetime.now(timezone.utc), session_id,
        )
        logger.info("session_summary_updated", session_id=session_id)

    async def delete_old_turns(self, session_id: str, keep_count: int = 10) -> int:
        """Delete old conversation turns, keeping the most recent `keep_count`."""
        result = await self._db.execute(
            """DELETE FROM conversations
               WHERE id IN (
                   SELECT id FROM conversations
                   WHERE session_id = $1::uuid
                   ORDER BY created_at DESC
                   OFFSET $2
               )""",
            session_id, keep_count,
        )
        count = 0
        if result:
            try:
                count = int(result.split()[-1]) if "DELETE" in result else 0
            except (ValueError, IndexError):
                count = 0
        return count

    # ── User Memory (长期记忆) ─────────────────────────────────

    async def save_user_memory(
        self,
        user_id: str,
        memory_type: str = "fact",
        content: str = "",
        session_id: str | None = None,
    ) -> str:
        """保存一条用户长期记忆."""
        row = await self._db.fetchrow(
            """INSERT INTO user_memories (user_id, memory_type, content, source_session_id)
               VALUES ($1, $2, $3, $4::uuid)
               RETURNING id""",
            user_id, memory_type, content, session_id,
        )
        memory_id = str(row["id"])
        logger.info("user_memory_saved", memory_id=memory_id, user_id=user_id, type=memory_type)
        return memory_id

    async def load_user_memories(
        self, user_id: str, limit: int = 20, session_id: str | None = None
    ) -> list[dict]:
        """加载用户长期记忆列表.

        When session_id is provided, returns memories scoped to that session
        plus legacy memories (source_session_id IS NULL) for backward compatibility.
        When session_id is None/omitted, returns all memories for the user.
        """
        if session_id:
            rows = await self._db.fetch(
                """SELECT id, user_id, memory_type, content, source_session_id, created_at, expires_at
                   FROM user_memories
                   WHERE user_id = $1
                     AND (source_session_id = $2::uuid OR source_session_id IS NULL)
                     AND (expires_at IS NULL OR expires_at > NOW())
                   ORDER BY created_at DESC
                   LIMIT $3""",
                user_id, session_id, limit,
            )
        else:
            rows = await self._db.fetch(
                """SELECT id, user_id, memory_type, content, source_session_id, created_at, expires_at
                   FROM user_memories
                   WHERE user_id = $1
                     AND (expires_at IS NULL OR expires_at > NOW())
                   ORDER BY created_at DESC
                   LIMIT $2""",
                user_id, limit,
            )
        return [
            {
                "id": str(r["id"]),
                "user_id": r["user_id"],
                "memory_type": r["memory_type"],
                "content": r["content"],
                "source_session_id": str(r["source_session_id"]) if r.get("source_session_id") else None,
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "expires_at": r["expires_at"].isoformat() if r.get("expires_at") else None,
            }
            for r in rows
        ]

    async def delete_user_memory(self, memory_id: str, user_id: str) -> bool:
        """删除用户的一条记忆."""
        result = await self._db.execute(
            "DELETE FROM user_memories WHERE id = $1::uuid AND user_id = $2",
            memory_id, user_id,
        )
        affected = "DELETE 1" in result if result else False
        if affected:
            logger.info("user_memory_deleted", memory_id=memory_id, user_id=user_id)
        return affected

    async def clear_user_memories(self, user_id: str) -> int:
        """清除用户的所有记忆."""
        result = await self._db.execute(
            "DELETE FROM user_memories WHERE user_id = $1", user_id,
        )
        try:
            count = int(result.split()[-1]) if result and "DELETE" in result else 0
        except (ValueError, IndexError):
            count = 0
        logger.info("user_memories_cleared", user_id=user_id, count=count)
        return count
