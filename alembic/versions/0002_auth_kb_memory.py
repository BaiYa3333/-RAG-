"""Add users_auth, knowledge_bases, kb_permissions, user_memories tables.

Revision ID: 0002_auth_kb_memory
Revises: 0001_function_expansion
Create Date: 2026-06-02
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_auth_kb_memory"
down_revision = "0001_function_expansion"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 用户认证表 ──────────────────────────────────────────
    op.create_table(
        "users_auth",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("username", sa.String(length=64), unique=True, nullable=False),
        sa.Column("email", sa.String(length=255), unique=True, nullable=True),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False, server_default="user"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # ── 知识库表 ────────────────────────────────────────────
    op.create_table(
        "knowledge_bases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("name", sa.String(length=128), unique=True, nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users_auth.id", ondelete="SET NULL"), nullable=True),
        sa.Column("is_public", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # ── 知识库权限表 ────────────────────────────────────────
    op.create_table(
        "kb_permissions",
        sa.Column("kb_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users_auth.id", ondelete="CASCADE"), nullable=False),
        sa.Column("permission", sa.String(length=20), nullable=False, server_default="read"),
        sa.PrimaryKeyConstraint("kb_id", "user_id"),
    )

    # ── 文档表新增 kb_id ────────────────────────────────────
    op.add_column("documents", sa.Column("kb_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("knowledge_bases.id", ondelete="SET NULL"), nullable=True))
    op.create_index("idx_documents_kb_id", "documents", ["kb_id"])

    # ── 用户长期记忆表 ──────────────────────────────────────
    op.create_table(
        "user_memories",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", sa.String(length=255), nullable=False, index=True),
        sa.Column("memory_type", sa.String(length=50), nullable=False, server_default="fact"),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_user_memories_user_id", "user_memories", ["user_id"])


def downgrade() -> None:
    op.drop_index("idx_user_memories_user_id", table_name="user_memories")
    op.drop_table("user_memories")
    op.drop_index("idx_documents_kb_id", table_name="documents")
    op.drop_column("documents", "kb_id")
    op.drop_table("kb_permissions")
    op.drop_table("knowledge_bases")
    op.drop_table("users_auth")
