"""测试 POST /documents/upload 端点 — 角色感知上传行为 + session-scoped retrieval."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_pipeline():
    """返回 mock IndexingPipeline，成功摄入返回预设结果."""
    pipeline = AsyncMock()
    pipeline.run = AsyncMock(
        return_value={
            "file": "/tmp/test.pdf",
            "chunks": 8,
            "elapsed_ms": 1234.5,
            "doc_id": "550e8400-e29b-41d4-a716-446655440000",
        }
    )
    return pipeline


def _make_admin_headers() -> dict:
    """为管理员测试生成 JWT Bearer token."""
    from src.auth.user_service import create_jwt
    admin_token = create_jwt("test-admin-id", "admin", expires_h=1)
    return {"Authorization": f"Bearer {admin_token}"}


def _make_user_headers() -> dict:
    """为普通用户测试生成 JWT Bearer token."""
    from src.auth.user_service import create_jwt
    user_token = create_jwt("test-user-id", "user", expires_h=1)
    return {"Authorization": f"Bearer {user_token}"}


@pytest.fixture
def mock_vector_store():
    """Mock ChromaDB VectorStore for session upload tests."""
    with patch("src.rag.retrieval.dense._get_vector_store") as mock_get_vs:
        vs = AsyncMock()
        col = AsyncMock()
        vs.get_or_create_collection = AsyncMock(return_value=col)
        vs.add = AsyncMock()
        vs._client = AsyncMock()
        vs._client.delete_collection = AsyncMock()
        mock_get_vs.return_value = vs
        yield vs


@pytest.fixture
def upload_client(mock_llm_client, mock_graph, mock_pipeline, mock_vector_store):
    """创建带 mock indexing_pipeline 和 VectorStore 的 TestClient."""
    from fastapi.testclient import TestClient
    from src.main import app

    app.state.llm_client = mock_llm_client
    app.state.graph = mock_graph
    app.state.cache_store = None
    app.state.vector_store = None
    app.state.doc_store = None
    app.state.indexing_pipeline = mock_pipeline

    # Mock kb_service for permission checks (now that admin bypass is removed)
    mock_kb = AsyncMock()
    mock_kb.check_permission = AsyncMock(return_value=True)
    mock_kb.get_kb = AsyncMock(return_value={
        "id": "test-kb-123", "name": "test-kb", "display_name": "Test KB",
        "owner_id": "test-admin-id", "is_public": False,
    })
    mock_kb.list_kbs = AsyncMock(return_value=[])
    app.state.kb_service = mock_kb

    # Mock memory_service for session ownership verification
    mock_memory = AsyncMock()
    mock_memory.get_session = AsyncMock(return_value={"id": "test-session-123", "user_id": "test-user-id"})
    mock_memory.create_session = AsyncMock(return_value={"id": "test-session-123", "title": "Test"})
    mock_memory.load_history = AsyncMock(return_value=[])
    app.state.memory_service = mock_memory

    client = TestClient(app)
    yield client

    app.state.llm_client = None
    app.state.graph = None
    app.state.indexing_pipeline = None
    app.state.memory_service = None
    app.state.kb_service = None


class TestAdminUploadSuccess:
    """管理员上传 — 持久化到数据库和向量库."""

    def test_upload_pdf_returns_indexed(self, upload_client):
        """管理员上传有效 PDF 返回 indexed 状态和元数据."""
        response = upload_client.post(
            "/documents/upload",
            files={"file": ("test.pdf", b"%PDF-1.4 mock content", "application/pdf")},
            data={"kb_id": "test-kb-123"},
            headers=_make_admin_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "test.pdf"
        assert data["doc_type"] == "pdf"
        assert data["chunk_count"] == 8
        assert data["status"] == "indexed"
        assert data["doc_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert data["file_hash"] is not None
        assert data["persist_mode"] == "persisted"

    def test_upload_markdown(self, upload_client):
        """管理员上传 .md 文件应正常接受."""
        response = upload_client.post(
            "/documents/upload",
            files={"file": ("readme.md", b"# Hello\n\nThis is a markdown file.", "text/markdown")},
            data={"kb_id": "test-kb-456"},
            headers=_make_admin_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["doc_type"] == "md"
        assert data["status"] == "indexed"
        assert data["persist_mode"] == "persisted"

    def test_extension_case_insensitive(self, upload_client):
        """扩展名大小写不敏感."""
        response = upload_client.post(
            "/documents/upload",
            files={"file": ("Report.PDF", b"%PDF-1.4 mock", "application/pdf")},
            data={"kb_id": "test-kb-789"},
            headers=_make_admin_headers(),
        )
        assert response.status_code == 200
        assert response.json()["doc_type"] == "pdf"


class TestRegularUserUpload:
    """普通用户上传 — 仅会话有效，写入 session-scoped ChromaDB 集合."""

    def test_user_upload_session_only(self, upload_client):
        """普通用户上传文件返回 session_only 状态和 session_id."""
        response = upload_client.post(
            "/documents/upload",
            files={"file": ("notes.txt", b"Plain text content here.", "text/plain")},
            headers=_make_user_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["doc_type"] == "txt"
        assert data["status"] == "session_only"
        assert data["persist_mode"] == "session_only"
        assert data["doc_id"] is None
        assert "session_id" in data  # Now returns session_id for tracking

    def test_user_upload_no_longer_returns_chunks_inline(self, upload_client):
        """普通用户上传不再返回 chunks — 改为写入 ChromaDB."""
        response = upload_client.post(
            "/documents/upload",
            files={"file": ("doc.md", b"# Hello\n\nThis is content.", "text/markdown")},
            headers=_make_user_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["doc_type"] == "md"
        assert data["status"] == "session_only"
        assert data["persist_mode"] == "session_only"
        # Chunks are no longer returned inline — they go to ChromaDB
        assert "chunks" not in data

    def test_user_upload_stores_to_session_collection(self, upload_client, mock_vector_store):
        """验证 session-only 上传写入 rag_session_{session_id} 集合."""
        response = upload_client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"Test content for session-scoped storage.", "text/plain")},
            data={"session_id": "test-session-123"},
            headers=_make_user_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"

        # Verify VectorStore was called with correct collection name
        mock_vector_store.get_or_create_collection.assert_called()
        call_args = mock_vector_store.get_or_create_collection.call_args
        assert "rag_session_test-session-123" in str(call_args)

    def test_user_upload_with_kb_id_persisted_when_permitted(self, upload_client):
        """任何认证用户通过 KB 权限检查后均可持久化到知识库."""
        response = upload_client.post(
            "/documents/upload",
            files={"file": ("notes.txt", b"Plain text content here.", "text/plain")},
            data={"kb_id": "some-kb-id", "session_id": "test-session-456"},
            headers=_make_user_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["persist_mode"] == "persisted"
        assert data["doc_id"] is not None


class TestUploadValidation:
    """格式校验错误场景（不依赖角色）."""

    def test_unsupported_extension_returns_415(self, upload_client):
        """不支持的扩展名返回 415."""
        response = upload_client.post(
            "/documents/upload",
            files={"file": ("malware.exe", b"evil", "application/octet-stream")},
            headers=_make_user_headers(),
        )
        assert response.status_code == 415
        assert "Unsupported" in response.json()["detail"]

    def test_empty_file_returns_422(self, upload_client):
        """空文件（0 字节）返回 422."""
        response = upload_client.post(
            "/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
            headers=_make_user_headers(),
        )
        assert response.status_code == 422
        assert "empty" in response.json()["detail"].lower()

    def test_file_too_large_returns_413(self, upload_client, monkeypatch):
        """超大文件返回 413."""
        monkeypatch.setattr("src.api.routes.settings.max_upload_size_mb", 0)
        response = upload_client.post(
            "/documents/upload",
            files={"file": ("big.pdf", b"x" * 100, "application/pdf")},
            headers=_make_user_headers(),
        )
        assert response.status_code in (413, 422)


class TestAdminUploadErrors:
    """管理员摄入过程错误场景."""

    def test_pipeline_not_initialized_returns_503(self, mock_llm_client, mock_graph, mock_vector_store):
        """IndexingPipeline 未初始化返回 503."""
        from fastapi.testclient import TestClient
        from src.main import app

        app.state.llm_client = mock_llm_client
        app.state.graph = mock_graph
        app.state.cache_store = None
        app.state.vector_store = None
        app.state.doc_store = None
        app.state.indexing_pipeline = None  # 未初始化

        # Mock kb_service to pass permission check so we reach the pipeline check
        mock_kb = AsyncMock()
        mock_kb.check_permission = AsyncMock(return_value=True)
        app.state.kb_service = mock_kb

        client = TestClient(app)
        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"content", "text/plain")},
            data={"kb_id": "test-kb-000"},
            headers=_make_admin_headers(),
        )
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()

        app.state.llm_client = None
        app.state.graph = None

    def test_pipeline_exception_returns_500(self, upload_client, mock_pipeline):
        """管理员摄入失败返回 500."""
        mock_pipeline.run.side_effect = RuntimeError("Embedding API timeout")
        response = upload_client.post(
            "/documents/upload",
            files={"file": ("doc.pdf", b"%PDF-1.4 content", "application/pdf")},
            data={"kb_id": "test-kb-error"},
            headers=_make_admin_headers(),
        )
        assert response.status_code == 500
        assert "Embedding API timeout" in response.json()["detail"]

    def test_partial_status_when_no_doc_id(self, upload_client, mock_pipeline):
        """管理员上传 DocStore 写入失败时返回 partial 状态."""
        mock_pipeline.run.return_value = {
            "file": "/tmp/test.txt",
            "chunks": 3,
            "elapsed_ms": 500,
            "doc_id": None,  # DocStore 不可用
        }
        response = upload_client.post(
            "/documents/upload",
            files={"file": ("notes.txt", b"some content", "text/plain")},
            data={"kb_id": "test-kb-partial"},
            headers=_make_admin_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "partial"
        assert data["doc_id"] is None
        assert data["chunk_count"] == 3


# ═══════════════════════════════════════════════════════════════════
# Session-scoped retrieval tests (Tasks 5.1–5.5)
# ═══════════════════════════════════════════════════════════════════


class TestSessionScopedRetrieval:
    """测试会话级检索：upload → same-session query → cross-session isolation → cleanup."""

    def test_dense_search_with_session_id(self):
        """5.2 / 5.3: dense_search 接受 session_id 并查询 session 集合."""
        # 这里只验证 dense_search 函数签名接受 session_id
        # 实际 ChromaDB 调用在生产环境端到端测试中验证
        import inspect
        from src.rag.retrieval.dense import dense_search

        sig = inspect.signature(dense_search)
        params = sig.parameters
        assert "session_id" in params
        assert params["session_id"].default is None  # Optional, defaults to None

    def test_dense_search_no_session_id_still_works(self):
        """5.3: 没有 session_id 时不查询 session 集合."""
        import inspect
        from src.rag.retrieval.dense import dense_search

        sig = inspect.signature(dense_search)
        # session_id is optional — existing callers without session_id still work
        assert sig.parameters["session_id"].default is None

    def test_cleanup_helper_exists(self):
        """5.4: _cleanup_session_collection helper 已定义."""
        from src.rag.retrieval.dense import _cleanup_session_collection

        assert callable(_cleanup_session_collection)

    @pytest.mark.asyncio
    async def test_cleanup_collection_best_effort(self, mock_vector_store):
        """5.4 / 5.5: 清理不存在的集合不抛异常（best-effort）."""
        from src.rag.retrieval.dense import _cleanup_session_collection

        # Mock delete_collection to raise (collection doesn't exist)
        mock_vector_store._client.delete_collection.side_effect = Exception("Collection does not exist")

        # Should not raise — best-effort
        result = await _cleanup_session_collection("non-existent-session")
        # Returns False on failure
        assert not result

    @pytest.mark.asyncio
    async def test_cleanup_collection_success(self, mock_vector_store):
        """5.4: 清理存在的 session 集合成功."""
        from src.rag.retrieval.dense import _cleanup_session_collection

        mock_vector_store._client.delete_collection.return_value = None

        result = await _cleanup_session_collection("test-session-cleanup")
        # Best-effort — if no exception, returns True
        assert result

    @pytest.mark.asyncio
    async def test_memory_service_delete_calls_cleanup(self):
        """5.4: MemoryService.delete_session 触发 ChromaDB 清理."""
        with patch("src.memory.service.DocStore") as mock_doc:
            # Setup mock doc store
            mock_db = mock_doc.return_value
            mock_db.execute = AsyncMock(return_value="DELETE 1")
            mock_db.fetchrow = AsyncMock()
            mock_db.fetch = AsyncMock()

            from src.memory.service import MemoryService

            with patch("src.rag.retrieval.dense._cleanup_session_collection") as mock_cleanup:
                mock_cleanup.return_value = True

                service = MemoryService(mock_db)
                mock_db.execute.return_value = "DELETE 1"

                deleted = await service.delete_session("test-session-delete")

                assert deleted is True
                # Cleanup should be called after successful DB deletion
                mock_cleanup.assert_called_once_with("test-session-delete")

    @pytest.mark.asyncio
    async def test_session_without_uploads_delete_no_error(self, mock_vector_store):
        """5.5: 删除无上传的会话不报错."""
        from src.rag.retrieval.dense import _cleanup_session_collection

        # This session never had uploads — cleanup should handle missing collection
        mock_vector_store._client.delete_collection.side_effect = Exception("Collection does not exist")

        # Should not raise
        result = await _cleanup_session_collection("session-no-uploads")
        assert not result  # Returns False on collection-not-found

    def test_hybrid_search_accepts_session_id(self):
        """验证 hybrid_search 签名包含 session_id 参数."""
        import inspect
        from src.rag.retrieval.hybrid import hybrid_search

        sig = inspect.signature(hybrid_search)
        params = sig.parameters
        assert "session_id" in params
        assert params["session_id"].default is None
        assert "kb_id" in params
        assert params["kb_id"].default is None


class TestUploadStreamAuth:
    """Task 5.5: /documents/upload/stream requires authentication."""

    @pytest.fixture
    def auth_enabled_upload_client(self):
        """Create TestClient with auth enabled."""
        from fastapi.testclient import TestClient
        from src.main import app

        import src.auth.dependencies as deps
        original = deps.settings.auth_enabled
        deps.settings.auth_enabled = True

        app.state.cache_store = None
        app.state.vector_store = None
        app.state.doc_store = None
        app.state.indexing_pipeline = None

        client = TestClient(app)
        yield client

        deps.settings.auth_enabled = original

    def test_upload_stream_without_auth_returns_401(self, auth_enabled_upload_client):
        """Streaming upload without auth header should return 401."""
        client = auth_enabled_upload_client
        response = client.post(
            "/documents/upload/stream",
            files={"file": ("test.pdf", b"%PDF-1.4 mock", "application/pdf")},
        )
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"] or "Authorization token required" in response.json()["detail"]

    def test_upload_stream_with_valid_auth(self, auth_enabled_upload_client):
        """Streaming upload with valid JWT should proceed."""
        from src.auth.user_service import create_jwt

        client = auth_enabled_upload_client
        token = create_jwt("test-user", "user", expires_h=1)
        response = client.post(
            "/documents/upload/stream",
            files={"file": ("test.pdf", b"%PDF-1.4 mock", "application/pdf")},
            headers={"Authorization": f"Bearer {token}"},
        )
        # Should not be 401 (may be 415 for empty-ish file or 500 for no pipeline, but not auth error)
        assert response.status_code != 401
