#!/bin/bash
# ── RAG_Project Docker Entrypoint ──────────────────────────
# 1. 等待 PostgreSQL 就绪
# 2. 运行 alembic migration（幂等，已执行过的跳过）
# 3. 启动 FastAPI 应用

set -e

echo "📦 RAG_Project starting..."
echo "⏳ Waiting for PostgreSQL ($POSTGRES_HOST)..."

# 优先使用 psql，不可用时降级为 Python socket 检查
wait_for_postgres() {
    if command -v psql &> /dev/null; then
        until PGPASSWORD="${PGPASSWORD:-changeme}" psql -h "${POSTGRES_HOST:-postgres}" -U raguser -d ragdb -c '\q' 2>/dev/null; do
            echo "   ...still waiting for DB (psql)"
            sleep 2
        done
    else
        echo "   (psql not available, using Python fallback)"
        until python -c "
import socket, time
try:
    s = socket.create_connection(('${POSTGRES_HOST:-postgres}', 5432), timeout=3)
    s.close()
    print('OK')
except Exception:
    exit(1)
" 2>/dev/null; do
            echo "   ...still waiting for DB (socket)"
            sleep 2
        done
    fi
}
wait_for_postgres
echo "✅ PostgreSQL is ready."

echo "📋 Running alembic migrations..."
alembic upgrade head
echo "✅ Alembic migrations complete."

echo "🚀 Starting uvicorn..."
exec python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
