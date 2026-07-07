"""SHA256 文件完整性校验 — SQLite 摄入历史（幂等跳过已摄入文件）."""

import os
import sqlite3
from datetime import datetime, timezone

from src.utils.logger import logger


DB_DIR = os.path.join("data", "db")
DB_PATH = os.path.join(DB_DIR, "ingestion_history.db")


class IngestionCache:
    """基于 SQLite 的摄入历史缓存 — 按文件 SHA-256 去重。"""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    # ── Schema ────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash   TEXT PRIMARY KEY,
                    file_path   TEXT NOT NULL,
                    file_size   INTEGER,
                    collection  TEXT,
                    chunk_count INTEGER,
                    status      TEXT CHECK(status IN ('success','failed','skipped')),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_message TEXT
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ingestion_collection
                ON ingestion_history(collection);
            """)

    # ── Connection helper ─────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Public API ────────────────────────────────────────

    def should_skip(self, file_hash: str) -> bool:
        """检查文件是否已成功摄入（可以被跳过）。

        Returns True 当记录存在且 status='success'。
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
        return row is not None and row["status"] == "success"

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        collection: str | None = None,
        chunk_count: int = 0,
    ) -> None:
        """记录成功摄入（INSERT OR REPLACE）。"""
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ingestion_history
                   (file_hash, file_path, file_size, collection, chunk_count, status, processed_at, error_message)
                   VALUES (?, ?, ?, ?, ?, 'success', ?, NULL)""",
                (file_hash, file_path, file_size, collection, chunk_count, _now()),
            )

    def mark_failed(self, file_hash: str, file_path: str, error: str) -> None:
        """记录失败摄入。"""
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ingestion_history
                   (file_hash, file_path, file_size, collection, chunk_count, status, processed_at, error_message)
                   VALUES (?, ?, ?, NULL, 0, 'failed', ?, ?)""",
                (file_hash, file_path, file_size, _now(), error),
            )

    def list_processed(self, collection: str | None = None) -> list[dict]:
        """列出已处理文件（按时间倒序），可选按 collection 过滤。"""
        with self._conn() as conn:
            if collection:
                rows = conn.execute(
                    """SELECT file_hash, file_path, file_size, collection, chunk_count,
                              status, processed_at, error_message
                       FROM ingestion_history
                       WHERE collection = ?
                       ORDER BY processed_at DESC""",
                    (collection,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT file_hash, file_path, file_size, collection, chunk_count,
                              status, processed_at, error_message
                       FROM ingestion_history
                       ORDER BY processed_at DESC""",
                ).fetchall()
        return [dict(r) for r in rows]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
