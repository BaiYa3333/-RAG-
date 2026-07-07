"""Memory data models — plain dataclasses mirroring the sessions/conversations tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Session:
    id: str
    user_id: str | None = None
    title: str | None = None
    summary: str | None = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row: dict) -> "Session":
        return cls(
            id=str(row["id"]),
            user_id=row.get("user_id"),
            title=row.get("title"),
            summary=row.get("summary"),
            metadata=row.get("metadata") or {},
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


@dataclass
class ConversationTurn:
    id: str
    session_id: str
    role: str  # user | assistant | system
    content: str
    metadata: dict = field(default_factory=dict)
    created_at: datetime | None = None

    @classmethod
    def from_row(cls, row: dict) -> "ConversationTurn":
        return cls(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            role=row["role"],
            content=row["content"],
            metadata=row.get("metadata") or {},
            created_at=row.get("created_at"),
        )
