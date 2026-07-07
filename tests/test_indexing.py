"""Unit tests for _sanitize_metadata and list_kbs SQL logic (Tasks 5.3, 5.4)."""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestSanitizeMetadata:
    """Task 5.4: _sanitize_metadata tests."""

    def test_nested_dict_removed(self):
        """Metadata with nested dict values should have those keys removed."""
        from src.rag.indexing.pipeline import _sanitize_metadata

        meta = {"source": "test.pdf", "parent_content": {"nested": "value"}}
        result = _sanitize_metadata(meta)
        assert result == {"source": "test.pdf"}
        assert "parent_content" not in result

    def test_list_values_removed(self):
        """Metadata with list values should have those keys removed."""
        from src.rag.indexing.pipeline import _sanitize_metadata

        meta = {"source": "test.pdf", "tags": ["tag1", "tag2"]}
        result = _sanitize_metadata(meta)
        assert result == {"source": "test.pdf"}
        assert "tags" not in result

    def test_primitive_values_preserved(self):
        """Primitive values (str, int, float, bool) should be kept."""
        from src.rag.indexing.pipeline import _sanitize_metadata

        meta = {
            "source": "test.pdf",
            "page": 1,
            "score": 0.95,
            "is_title": True,
        }
        result = _sanitize_metadata(meta)
        assert result == meta

    def test_none_values_removed(self):
        """None values should be filtered out."""
        from src.rag.indexing.pipeline import _sanitize_metadata

        meta = {"source": "test.pdf", "optional": None}
        result = _sanitize_metadata(meta)
        assert result == {"source": "test.pdf"}
        assert "optional" not in result

    def test_mixed_values(self):
        """Mixed primitive and non-primitive: only primitives kept."""
        from src.rag.indexing.pipeline import _sanitize_metadata

        meta = {
            "source": "test.pdf",
            "page": 1,
            "extra": {"nested": True},
            "tags": ["a", "b"],
        }
        result = _sanitize_metadata(meta)
        assert "source" in result
        assert "page" in result
        assert "extra" not in result
        assert "tags" not in result

    def test_empty_metadata(self):
        """Empty metadata should return empty dict."""
        from src.rag.indexing.pipeline import _sanitize_metadata

        assert _sanitize_metadata({}) == {}


class TestListKbsSQL:
    """Task 5.3: list_kbs SQL construction tests — verify correct WHERE clauses
    for all four argument combinations."""

    @pytest.fixture
    def mock_doc_store(self):
        """Create a mock DocStore that captures the SQL query."""
        store = MagicMock()
        store.fetchrow = AsyncMock()
        store.fetch = AsyncMock(return_value=[])
        store.execute = AsyncMock()
        return store

    @pytest.fixture
    def kb_service(self, mock_doc_store):
        from src.rag.knowledge_base.service import KnowledgeBaseService
        return KnowledgeBaseService(mock_doc_store)

    @pytest.mark.asyncio
    async def test_include_public_true_with_user_id(self, kb_service, mock_doc_store):
        """include_public=True, user_id=X → public OR owned_by_X OR has_permission_X."""
        await kb_service.list_kbs(user_id="user-1", include_public=True)
        query = mock_doc_store.fetch.call_args[0][0]
        # Should contain OR combining public with ownership/permission check
        assert "kb.is_public = true" in query
        assert "kb.owner_id = $1::uuid" in query
        assert "kb_permissions" in query

    @pytest.mark.asyncio
    async def test_include_public_true_without_user_id(self, kb_service, mock_doc_store):
        """include_public=True, user_id=None → only is_public condition."""
        await kb_service.list_kbs(user_id=None, include_public=True)
        query = mock_doc_store.fetch.call_args[0][0]
        assert "kb.is_public = true" in query
        assert "kb.owner_id" not in query  # No user filter

    @pytest.mark.asyncio
    async def test_include_public_false_with_user_id(self, kb_service, mock_doc_store):
        """include_public=False, user_id=X → owned_by_X OR has_permission_X."""
        await kb_service.list_kbs(user_id="user-1", include_public=False)
        query = mock_doc_store.fetch.call_args[0][0]
        # Should have owner/permission filter but NOT is_public
        assert "kb.owner_id = $1::uuid" in query
        assert "kb_permissions" in query
        # Both includes should be absent since include_public=False
        assert "kb.is_public" not in query

    @pytest.mark.asyncio
    async def test_include_public_false_without_user_id(self, kb_service, mock_doc_store):
        """include_public=False, user_id=None → no WHERE filter on the main query (all KBs)."""
        await kb_service.list_kbs(user_id=None, include_public=False)
        query = mock_doc_store.fetch.call_args[0][0]
        # Neither $1 parameter nor is_public should appear in the main WHERE
        assert "$1" not in query
        # "kb.is_public" should not appear outside the subquery
        main_query = query.split("FROM knowledge_bases kb")[1]
        assert "kb.is_public" not in main_query
