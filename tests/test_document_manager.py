"""Unit tests for DocumentManager — Task 8.4."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDocumentManager:
    """Tests for DocumentManager: single doc delete, KB delete, partial failure."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all backend dependencies."""
        # Mock VectorStore
        mock_vs = AsyncMock()
        mock_col = AsyncMock()
        mock_vs.get_or_create_collection = AsyncMock(return_value=mock_col)
        mock_vs.delete_where = AsyncMock()
        mock_vs._client = MagicMock()
        mock_vs._client.delete_collection = AsyncMock()

        # Mock DocStore
        mock_ds = AsyncMock()
        mock_ds.execute = AsyncMock()

        return {
            "vector_store": mock_vs,
            "doc_store": mock_ds,
            "collection": mock_col,
        }

    @pytest.mark.asyncio
    async def test_delete_document_all_success(self, mock_dependencies):
        """正常单文档删除：所有后端成功."""
        from src.rag.document_manager import DocumentManager

        with patch("src.rag.document_manager.DocStore", return_value=mock_dependencies["doc_store"]), \
             patch("src.rag.retrieval.dense._get_vector_store",
                   AsyncMock(return_value=mock_dependencies["vector_store"])), \
             patch("src.graph.nodes.retrieval.invalidate_sparse_index", AsyncMock()), \
             patch("src.rag.ingestion.integrity.IngestionCache", MagicMock()), \
             patch("src.config.settings.ingestion_integrity_enabled", False):

            dm = DocumentManager(doc_store=mock_dependencies["doc_store"])
            results = await dm.delete_document("doc-123", kb_id="kb-456")

        assert results["chromadb"]["status"] == "success"
        assert results["postgresql"]["status"] == "success"
        assert results["bm25"]["status"] == "success"
        # ingestion_cache may show success even when integrity is disabled
        assert results["ingestion_cache"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_delete_document_partial_failure(self, mock_dependencies):
        """ChromaDB 失败时，其他后端应继续执行并返回错误详情."""
        from src.rag.document_manager import DocumentManager

        mock_dependencies["vector_store"].delete_where = AsyncMock(
            side_effect=Exception("ChromaDB connection refused")
        )

        with patch("src.rag.document_manager.DocStore", return_value=mock_dependencies["doc_store"]), \
             patch("src.rag.retrieval.dense._get_vector_store",
                   AsyncMock(return_value=mock_dependencies["vector_store"])), \
             patch("src.graph.nodes.retrieval.invalidate_sparse_index", AsyncMock()), \
             patch("src.rag.ingestion.integrity.IngestionCache", MagicMock()), \
             patch("src.config.settings.ingestion_integrity_enabled", False):

            dm = DocumentManager(doc_store=mock_dependencies["doc_store"])
            results = await dm.delete_document("doc-123", kb_id="kb-456")

        assert results["chromadb"]["status"] == "error"
        assert "ChromaDB connection refused" in results["chromadb"]["error"]
        # 其他后端应成功
        assert results["postgresql"]["status"] == "success"
        assert results["bm25"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_delete_kb_all_success(self, mock_dependencies):
        """正常 KB 级删除：所有后端成功."""
        from src.rag.document_manager import DocumentManager

        with patch("src.rag.document_manager.DocStore", return_value=mock_dependencies["doc_store"]), \
             patch("src.rag.retrieval.dense._get_vector_store",
                   AsyncMock(return_value=mock_dependencies["vector_store"])), \
             patch("src.graph.nodes.retrieval.invalidate_sparse_index", AsyncMock()), \
             patch("src.rag.ingestion.integrity.IngestionCache", MagicMock()), \
             patch("src.config.settings.ingestion_integrity_enabled", False):

            dm = DocumentManager(doc_store=mock_dependencies["doc_store"])
            results = await dm.delete_kb("kb-456")

        assert results["chromadb"]["status"] == "success"
        assert results["postgresql"]["status"] == "success"
        assert results["bm25"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_delete_kb_chromadb_failure(self, mock_dependencies):
        """ChromaDB collection 删除失败时仍继续处理其他后端."""
        from src.rag.document_manager import DocumentManager

        mock_dependencies["vector_store"]._client.delete_collection = AsyncMock(
            side_effect=Exception("Collection not found")
        )

        with patch("src.rag.document_manager.DocStore", return_value=mock_dependencies["doc_store"]), \
             patch("src.rag.retrieval.dense._get_vector_store",
                   AsyncMock(return_value=mock_dependencies["vector_store"])), \
             patch("src.graph.nodes.retrieval.invalidate_sparse_index", AsyncMock()), \
             patch("src.rag.ingestion.integrity.IngestionCache", MagicMock()), \
             patch("src.config.settings.ingestion_integrity_enabled", False):

            dm = DocumentManager(doc_store=mock_dependencies["doc_store"])
            results = await dm.delete_kb("kb-456")

        assert results["chromadb"]["status"] == "error"
        assert results["postgresql"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_delete_document_without_kb_id(self, mock_dependencies):
        """不提供 kb_id 时使用默认 collection."""
        from src.rag.document_manager import DocumentManager

        with patch("src.rag.document_manager.DocStore", return_value=mock_dependencies["doc_store"]), \
             patch("src.rag.retrieval.dense._get_vector_store",
                   AsyncMock(return_value=mock_dependencies["vector_store"])), \
             patch("src.graph.nodes.retrieval.invalidate_sparse_index", AsyncMock()), \
             patch("src.rag.ingestion.integrity.IngestionCache", MagicMock()), \
             patch("src.config.settings.ingestion_integrity_enabled", False):

            dm = DocumentManager(doc_store=mock_dependencies["doc_store"])
            results = await dm.delete_document("doc-123")

        # 应使用默认 collection "rag_docs_dev"
        assert results["chromadb"]["status"] == "success"
