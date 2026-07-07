-- ============================================================
-- RAG_Project PostgreSQL Init
-- docker-compose 首次启动时自动执行
-- 仅创建扩展，表结构由 alembic migration 在 entrypoint 中处理
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
