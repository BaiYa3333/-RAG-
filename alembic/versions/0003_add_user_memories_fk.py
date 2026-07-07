"""Add FK constraint on user_memories.source_session_id → sessions.id CASCADE.

Revision ID: 0003_add_user_memories_fk
Revises: 0002_auth_kb_memory
Create Date: 2026-06-02
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0003_add_user_memories_fk"
down_revision = "0002_auth_kb_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Clean dangling references: delete user_memories rows where
    #    source_session_id points to a non-existent session
    op.execute("""
        DELETE FROM user_memories
        WHERE source_session_id IS NOT NULL
          AND source_session_id NOT IN (SELECT id FROM sessions)
    """)

    # 2. Add foreign key constraint with CASCADE delete
    op.create_foreign_key(
        "fk_user_memories_source_session_id",
        "user_memories",
        "sessions",
        ["source_session_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_user_memories_source_session_id",
        "user_memories",
        type_="foreignkey",
    )
