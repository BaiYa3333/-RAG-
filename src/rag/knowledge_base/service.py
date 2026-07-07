"""多知识库管理服务."""

from __future__ import annotations

from datetime import datetime, timezone

from src.stores.doc_store import DocStore
from src.utils.logger import logger


class KnowledgeBaseService:
    """知识库 CRUD + 权限管理."""

    def __init__(self, doc_store: DocStore):
        self._db = doc_store

    # ── 知识库 CRUD ──────────────────────────────────────────

    async def create_kb(
        self,
        name: str,
        display_name: str,
        description: str | None = None,
        owner_id: str | None = None,
        is_public: bool = False,
    ) -> dict:
        """创建知识库."""
        row = await self._db.fetchrow(
            """INSERT INTO knowledge_bases (name, display_name, description, owner_id, is_public)
               VALUES ($1, $2, $3, $4, $5)
               RETURNING id, name, display_name, description, owner_id, is_public, created_at, updated_at""",
            name, display_name, description, owner_id, is_public,
        )
        kb_id = str(row["id"])
        logger.info("kb_created", kb_id=kb_id, name=name)
        return {
            "id": kb_id,
            "name": row["name"],
            "display_name": row["display_name"],
            "description": row.get("description"),
            "owner_id": str(row["owner_id"]) if row.get("owner_id") else None,
            "is_public": row["is_public"],
            "doc_count": 0,
            "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
            "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        }

    async def list_kbs(self, user_id: str | None = None, include_public: bool = True) -> list[dict]:
        """列出用户可访问的知识库。

        Constructs WHERE clause explicitly for each of the four argument combinations:
        - include_public=True, user_id=X  → public OR owned_by_X OR has_permission_X
        - include_public=True, user_id=None → public only
        - include_public=False, user_id=X  → owned_by_X OR has_permission_X
        - include_public=False, user_id=None → all KBs (no filter)
        """
        params: list = []
        where = ""

        if include_public and user_id:
            where = (
                "WHERE (kb.is_public = true OR kb.owner_id = $1::uuid OR "
                "EXISTS (SELECT 1 FROM kb_permissions WHERE kb_id = kb.id AND user_id = $1::uuid))"
            )
            params.append(user_id)
        elif include_public and not user_id:
            where = "WHERE kb.is_public = true"
        elif not include_public and user_id:
            where = (
                "WHERE (kb.owner_id = $1::uuid OR "
                "EXISTS (SELECT 1 FROM kb_permissions WHERE kb_id = kb.id AND user_id = $1::uuid))"
            )
            params.append(user_id)
        # else: both False/None → no filter, return all KBs

        query = f"""SELECT kb.*,
                     (SELECT COUNT(*) FROM documents d WHERE d.kb_id = kb.id) AS doc_count
                   FROM knowledge_bases kb
                   {where}
                   ORDER BY kb.updated_at DESC"""
        rows = await self._db.fetch(query, *params)
        return [
            {
                "id": str(r["id"]),
                "name": r["name"],
                "display_name": r["display_name"],
                "description": r.get("description"),
                "owner_id": str(r["owner_id"]) if r.get("owner_id") else None,
                "is_public": r["is_public"],
                "doc_count": r.get("doc_count", 0),
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
            }
            for r in rows
        ]

    async def get_kb(self, kb_id: str) -> dict | None:
        """获取知识库详情."""
        row = await self._db.fetchrow(
            """SELECT kb.*,
               (SELECT COUNT(*) FROM documents d WHERE d.kb_id = kb.id) AS doc_count
               FROM knowledge_bases kb
               WHERE kb.id = $1::uuid""",
            kb_id,
        )
        if not row:
            return None
        return {
            "id": str(row["id"]),
            "name": row["name"],
            "display_name": row["display_name"],
            "description": row.get("description"),
            "owner_id": str(row["owner_id"]) if row.get("owner_id") else None,
            "is_public": row["is_public"],
            "doc_count": row.get("doc_count", 0),
            "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
            "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        }

    async def delete_kb(self, kb_id: str) -> bool:
        """删除知识库（同时清理关联的 ChromaDB collection）。"""
        # 获取 kb name 用于清理 ChromaDB
        row = await self._db.fetchrow(
            "SELECT name FROM knowledge_bases WHERE id = $1::uuid", kb_id
        )
        if not row:
            return False

        # 尝试清理 ChromaDB collection（直接删除 collection，不创建）
        from src.rag.retrieval.collections import kb_collection_name
        col_name = kb_collection_name(str(kb_id))
        try:
            from src.rag.retrieval.dense import _get_vector_store
            vs = await _get_vector_store()
            if vs._client is not None:
                await vs.delete_collection(col_name)
                logger.info("kb_chroma_deleted", kb_id=kb_id, collection=col_name)
            else:
                logger.warning("kb_chroma_delete_skipped", kb_id=kb_id, reason="VectorStore not connected")
        except Exception as e:
            # Collection may not exist — that's fine for idempotent delete
            logger.warning("kb_chroma_delete_failed", kb_id=kb_id, collection=col_name, error=str(e))

        # 先删除关联的文档和权限记录（解除外键约束）
        await self._db.execute(
            "DELETE FROM documents WHERE kb_id = $1::uuid", kb_id
        )
        await self._db.execute(
            "DELETE FROM kb_permissions WHERE kb_id = $1::uuid", kb_id
        )
        await self._db.execute(
            "DELETE FROM knowledge_bases WHERE id = $1::uuid", kb_id
        )
        logger.info("kb_deleted", kb_id=kb_id)
        return True

    async def check_permission(self, kb_id: str, user_id: str, required: str = "read") -> bool:
        """检查用户对知识库的权限."""
        row = await self._db.fetchrow(
            """SELECT is_public, owner_id FROM knowledge_bases WHERE id = $1::uuid""",
            kb_id,
        )
        if not row:
            return False
        if row["is_public"]:
            return True
        if str(row["owner_id"]) == user_id:
            return True

        perm = await self._db.fetchrow(
            "SELECT permission FROM kb_permissions WHERE kb_id = $1::uuid AND user_id = $2::uuid",
            kb_id, user_id,
        )
        if not perm:
            return False
        if required == "read":
            return perm["permission"] in ("read", "write", "admin")
        if required == "write":
            return perm["permission"] in ("write", "admin")
        if required == "admin":
            return perm["permission"] == "admin"
        return False

    async def set_permission(self, kb_id: str, user_id: str, permission: str) -> None:
        """设置用户对知识库的权限."""
        await self._db.execute(
            """INSERT INTO kb_permissions (kb_id, user_id, permission)
               VALUES ($1::uuid, $2::uuid, $3)
               ON CONFLICT (kb_id, user_id)
               DO UPDATE SET permission = $3""",
            kb_id, user_id, permission,
        )
        logger.info("kb_permission_set", kb_id=kb_id, user_id=user_id, permission=permission)

    async def get_kb_stats(self, kb_id: str) -> dict:
        """获取知识库统计信息."""
        row = await self._db.fetchrow(
            """SELECT
                 COUNT(*)::int AS doc_count,
                 COALESCE(SUM(chunk_count), 0)::int AS total_chunks,
                 MAX(updated_at) AS last_updated
               FROM documents WHERE kb_id = $1::uuid""",
            kb_id,
        )
        return {
            "doc_count": row.get("doc_count", 0) if row else 0,
            "total_chunks": row.get("total_chunks", 0) if row else 0,
            "last_updated": row["last_updated"].isoformat() if row and row.get("last_updated") else None,
        }

    # ── 默认知识库 ──────────────────────────────────────────

    async def ensure_default_kb(self) -> dict:
        """确保默认知识库存在."""
        row = await self._db.fetchrow(
            "SELECT * FROM knowledge_bases WHERE name = $1", "default"
        )
        if row:
            return {
                "id": str(row["id"]),
                "name": row["name"],
                "display_name": row["display_name"],
            }

        return await self.create_kb(
            name="default",
            display_name="默认知识库",
            description="系统默认知识库（迁移自 rag_docs_dev）",
            is_public=True,
        )
