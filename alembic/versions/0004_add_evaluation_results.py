"""Add evaluation_results table for RAGAS evaluation persistence.

Revision ID: 0004_add_evaluation_results
Revises: 0003_add_user_memories_fk
Create Date: 2026-06-03
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_add_evaluation_results"
down_revision = "0003_add_user_memories_fk"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("testset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("config_label", sa.String(length=128), nullable=False, server_default="ragas_eval"),
        sa.Column("faithfulness", sa.Float(), nullable=False, server_default="0"),
        sa.Column("answer_relevancy", sa.Float(), nullable=False, server_default="0"),
        sa.Column("context_precision", sa.Float(), nullable=False, server_default="0"),
        sa.Column("context_recall", sa.Float(), nullable=False, server_default="0"),
        sa.Column("config_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("evaluation_results")
