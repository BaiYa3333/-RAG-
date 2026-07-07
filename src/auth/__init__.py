"""Auth data models — simple dataclasses for the auth subsystem.

UserRole is retained for backward compatibility only.
The admin/user role distinction has been removed;
all authenticated users have equal permissions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    admin = "admin"
    user = "user"
    viewer = "viewer"


@dataclass
class ApiKey:
    id: str
    user_id: str
    role: str = "user"
    label: str | None = None
    key_hash: str = ""
    revoked: bool = False
    created_at: datetime | None = None
    last_used: datetime | None = None

    @classmethod
    def from_row(cls, row: dict) -> "ApiKey":
        return cls(
            id=str(row["id"]),
            user_id=row["user_id"],
            role=row.get("role", "user"),
            label=row.get("label"),
            key_hash=row.get("key_hash", ""),
            revoked=row.get("revoked", False),
            created_at=row.get("created_at"),
            last_used=row.get("last_used"),
        )


@dataclass
class User:
    user_id: str
    role: str = "user"
    api_key_id: str | None = None
