"""统一文档生命周期管理 — 协调 ChromaDB / PostgreSQL / BM25 / IngestionCache 的删除操作.

DocumentManager 提供统一的删除接口，确保跨存储的一致性：
- delete_document: 删除单个文档（ChromaDB 向量 + PostgreSQL 记录 + BM25 索引更新 + IngestionCache 标记）
- delete_kb: 删除整个知识库（Drop collection + 删除所有记录 + 清除 BM25 文件 + 清除缓存）

所有操作采用部分失败收集模式：每个后端独立执行，收集结果后统一返回。
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.observability.langfuse_context import langfuse_context
from src.rag.retrieval.collections import kb_collection_name
from src.stores.doc_store import DocStore
from src.utils.logger import logger as structlog_logger

logger = logging.getLogger(__name__)


class DocumentManager:
    """统一文档生命周期管理器.

    协调多存储后端的文档/KB 删除操作。
    所有删除方法返回 {backend: status, error?: str} 字典。
    """

    def __init__(
        self,
        doc_store: DocStore | None = None,
    ):
        self._doc_store = doc_store

    async def _get_doc_store(self) -> DocStore:
        if self._doc_store is None:
            self._doc_store = DocStore()
            try:
                await self._doc_store.connect()
            except Exception as exc:
                logger.warning("document_manager_doc_store_connect_failed", error=str(exc))
        return self._doc_store

    # ── Public API ───────────────────────────────────────────

    async def delete_document(
        self, doc_id: str, kb_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """删除单个文档 — 协调所有存储后端的删除。

        Args:
            doc_id: 文档 ID
            kb_id: 知识库 ID（可选，用于 ChromaDB collection 定位）

        Returns:
            {backend: {"status": "success"|"error", "error": str|None}} 字典
        """
        results: dict[str, dict[str, Any]] = {}

        # ── 1. ChromaDB 向量删除 ──
        col_name = kb_collection_name(str(kb_id)) if kb_id else "rag_docs_dev"
        try:
            from src.rag.retrieval.dense import _get_vector_store
            vs = await _get_vector_store()
            col = await vs.get_or_create_collection(col_name)
            await vs.delete_where(col, {"doc_id": str(doc_id)})
            results["chromadb"] = {"status": "success"}
            structlog_logger.info(
                "document_manager_vector_deleted",
                doc_id=doc_id, kb_id=kb_id, collection=col_name,
            )
        except Exception as exc:
            results["chromadb"] = {"status": "error", "error": str(exc)}
            structlog_logger.warning(
                "document_manager_vector_delete_failed",
                doc_id=doc_id, kb_id=kb_id, error=str(exc),
            )

        # ── 2. PostgreSQL 记录删除 ──
        try:
            doc_store = await self._get_doc_store()
            await doc_store.execute(
                "DELETE FROM documents WHERE id = $1::uuid",
                doc_id,
            )
            if kb_id:
                await doc_store.execute(
                    "DELETE FROM documents WHERE id = $1::uuid AND kb_id = $2::uuid",
                    doc_id, kb_id,
                )
            results["postgresql"] = {"status": "success"}
            structlog_logger.info(
                "document_manager_postgres_deleted", doc_id=doc_id, kb_id=kb_id,
            )
        except Exception as exc:
            results["postgresql"] = {"status": "error", "error": str(exc)}
            structlog_logger.warning(
                "document_manager_postgres_delete_failed",
                doc_id=doc_id, kb_id=kb_id, error=str(exc),
            )

        # ── 3. BM25 索引更新 ──
        try:
            from src.graph.nodes.retrieval import invalidate_sparse_index
            await invalidate_sparse_index(col_name)
            results["bm25"] = {"status": "success"}
        except Exception as exc:
            results["bm25"] = {"status": "error", "error": str(exc)}
            structlog_logger.warning(
                "document_manager_bm25_invalidate_failed",
                doc_id=doc_id, kb_id=kb_id, error=str(exc),
            )

        # ── 4. IngestionCache 标记 ──
        try:
            if settings.ingestion_integrity_enabled:
                from src.rag.ingestion.integrity import IngestionCache
                cache = IngestionCache()
                # 直接将关联的条目状态更新为 'skipped'，允许重新摄入
                with cache._conn() as conn:
                    conn.execute(
                        "UPDATE ingestion_history SET status = 'skipped', error_message = ? "
                        "WHERE collection = ? AND status = 'success'",
                        (f"Document deleted: {doc_id}", col_name),
                    )
            results["ingestion_cache"] = {"status": "success"}
        except Exception as exc:
            results["ingestion_cache"] = {"status": "error", "error": str(exc)}
            structlog_logger.warning(
                "document_manager_cache_invalidate_failed",
                doc_id=doc_id, kb_id=kb_id, error=str(exc),
            )

        # Langfuse tracing
        try:
            langfuse_context.update_current_observation(
                operation="delete_document",
                doc_id=doc_id,
                kb_id=kb_id,
                results={k: v["status"] for k, v in results.items()},
            )
        except Exception:
            pass

        return results

    async def delete_kb(self, kb_id: str) -> dict[str, dict[str, Any]]:
        """删除整个知识库 — 协调所有存储后端的级联删除。

        Args:
            kb_id: 知识库 ID

        Returns:
            {backend: {"status": "success"|"error", "error": str|None}} 字典
        """
        results: dict[str, dict[str, Any]] = {}
        col_name = kb_collection_name(str(kb_id))

        # ── 1. ChromaDB collection 删除 ──
        try:
            from src.rag.retrieval.dense import _get_vector_store
            vs = await _get_vector_store()
            if vs._client is not None:
                await vs._client.delete_collection(col_name)
            results["chromadb"] = {"status": "success"}
            structlog_logger.info(
                "document_manager_kb_collection_deleted", kb_id=kb_id, collection=col_name,
            )
        except Exception as exc:
            results["chromadb"] = {"status": "error", "error": str(exc)}
            structlog_logger.warning(
                "document_manager_kb_collection_delete_failed", kb_id=kb_id, error=str(exc),
            )

        # ── 2. PostgreSQL 记录删除 ──
        try:
            doc_store = await self._get_doc_store()
            await doc_store.execute(
                "DELETE FROM documents WHERE kb_id = $1::uuid", kb_id,
            )
            results["postgresql"] = {"status": "success"}
            structlog_logger.info(
                "document_manager_kb_postgres_deleted", kb_id=kb_id,
            )
        except Exception as exc:
            results["postgresql"] = {"status": "error", "error": str(exc)}
            structlog_logger.warning(
                "document_manager_kb_postgres_delete_failed", kb_id=kb_id, error=str(exc),
            )

        # ── 3. BM25 索引文件删除 ──
        try:
            from src.graph.nodes.retrieval import invalidate_sparse_index
            await invalidate_sparse_index(col_name)
            results["bm25"] = {"status": "success"}
        except Exception as exc:
            results["bm25"] = {"status": "error", "error": str(exc)}
            structlog_logger.warning(
                "document_manager_kb_bm25_delete_failed", kb_id=kb_id, error=str(exc),
            )

        # ── 4. IngestionCache 条目清除 ──
        try:
            if settings.ingestion_integrity_enabled:
                from src.rag.ingestion.integrity import IngestionCache
                cache = IngestionCache()
                with cache._conn() as conn:
                    conn.execute(
                        "DELETE FROM ingestion_history WHERE collection = ?",
                        (col_name,),
                    )
            results["ingestion_cache"] = {"status": "success"}
        except Exception as exc:
            results["ingestion_cache"] = {"status": "error", "error": str(exc)}
            structlog_logger.warning(
                "document_manager_kb_cache_delete_failed", kb_id=kb_id, error=str(exc),
            )

        # Langfuse tracing
        try:
            langfuse_context.update_current_observation(
                operation="delete_kb",
                kb_id=kb_id,
                results={k: v["status"] for k, v in results.items()},
            )
        except Exception:
            pass

        return results
