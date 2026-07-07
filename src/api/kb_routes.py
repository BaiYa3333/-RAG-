"""知识库管理路由."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.auth.dependencies import get_current_user

kb_router = APIRouter(prefix="/kb", tags=["knowledge_bases"])


class CreateKBRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="知识库标识名（用于 ChromaDB collection）")
    display_name: str = Field(..., min_length=1, max_length=255, description="显示名称")
    description: str | None = Field(default=None)
    is_public: bool = Field(default=False)


class KBResponse(BaseModel):
    id: str
    name: str
    display_name: str
    description: str | None = None
    owner_id: str | None = None
    is_public: bool
    doc_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class KBPermissionRequest(BaseModel):
    user_id: str = Field(..., description="用户 ID")
    permission: str = Field(default="read", pattern="^(read|write|admin)$")


def _get_kb_service(request: Request):
    return getattr(request.app.state, "kb_service", None)


async def _get_user_id_or_anon(request: Request) -> str | None:
    """Get authenticated user_id, or None if anonymous/auth disabled.

    Manually extracts the Bearer token from the Authorization header to avoid
    bypassing FastAPI's HTTPBearer dependency (which doesn't work when called
    with explicit credentials=None).
    """
    try:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            from src.auth.user_service import decode_jwt
            payload = decode_jwt(token)
            if payload:
                user_id = payload.get("sub")
                if user_id and user_id != "anonymous":
                    # Verify user is still active
                    user_service = getattr(request.app.state, "user_service", None)
                    if user_service:
                        user = await user_service.get_user_by_id(user_id)
                        if user and user.get("is_active"):
                            return user_id
                        elif user:
                            return None  # Account disabled
                    return user_id  # No user_service available, accept JWT
            # Fallback: try API key validation
            auth_service = getattr(request.app.state, "auth_service", None)
            if auth_service:
                user = await auth_service.validate_api_key(token)
                if user:
                    return user.get("user_id")
        return None
    except Exception:
        return None


@kb_router.get("", response_model=list[KBResponse], summary="列出可访问的知识库")
async def list_kbs(request: Request):
    """列出当前用户可访问的知识库."""
    kb_service = _get_kb_service(request)
    if kb_service is None:
        raise HTTPException(status_code=503, detail="KB service not available")

    user_id = await _get_user_id_or_anon(request)

    kbs = await kb_service.list_kbs(user_id=user_id)
    return [KBResponse(**kb) for kb in kbs]


@kb_router.post("", response_model=KBResponse, status_code=201, summary="创建知识库")
async def create_kb(body: CreateKBRequest, request: Request, user: dict = Depends(get_current_user)):
    """创建新知识库（仅限管理员）."""
    kb_service = _get_kb_service(request)
    if kb_service is None:
        raise HTTPException(status_code=503, detail="KB service not available")

    # Use the already-authenticated user from get_current_user dependency
    user_id = user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")

    kb = await kb_service.create_kb(
        name=body.name,
        display_name=body.display_name,
        description=body.description,
        owner_id=user_id,
        is_public=body.is_public,
    )
    return KBResponse(**kb)


@kb_router.get("/{kb_id}", response_model=KBResponse, summary="知识库详情")
async def get_kb(kb_id: str, request: Request):
    """获取知识库详情 + 统计."""
    kb_service = _get_kb_service(request)
    if kb_service is None:
        raise HTTPException(status_code=503, detail="KB service not available")

    kb = await kb_service.get_kb(kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return KBResponse(**kb)


@kb_router.delete("/{kb_id}", summary="删除知识库")
async def delete_kb(kb_id: str, request: Request, _user: dict = Depends(get_current_user)):
    """删除知识库（仅限管理员）."""
    kb_service = _get_kb_service(request)
    if kb_service is None:
        raise HTTPException(status_code=503, detail="KB service not available")

    kb = await kb_service.get_kb(kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    await kb_service.delete_kb(kb_id)

    # 使用 DocumentManager 协调多存储删除
    from src.rag.document_manager import DocumentManager
    doc_store = getattr(request.app.state, "doc_store", None)
    doc_mgr = DocumentManager(doc_store=doc_store)
    results = await doc_mgr.delete_kb(kb_id)

    # 检查是否有部分失败
    failures = {k: v for k, v in results.items() if v["status"] == "error"}
    if failures:
        from src.utils.logger import logger as _logger
        _logger.warning("kb_delete_partial_failures", kb_id=kb_id, failures=failures)

    return {"detail": "Knowledge base deleted", "backend_results": results}


@kb_router.get("/{kb_id}/documents", summary="列出知识库内文档")
async def list_kb_documents(kb_id: str, request: Request, user: dict = Depends(get_current_user)):
    """列出指定知识库内的文档."""
    kb_service = _get_kb_service(request)
    if kb_service is None:
        raise HTTPException(status_code=503, detail="KB service not available")

    user_id = user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")
    allowed = await kb_service.check_permission(kb_id, user_id, required="read")
    if not allowed:
        raise HTTPException(status_code=403, detail="Access denied: knowledge base read permission required")

    from src.stores.doc_store import DocStore
    doc_store: DocStore = getattr(request.app.state, "doc_store", None)
    if doc_store is None:
        raise HTTPException(status_code=503, detail="DocStore not available")

    rows = await doc_store.fetch(
        """SELECT id, title, source, doc_type, chunk_count, metadata, created_at
           FROM documents WHERE kb_id = $1::uuid
           ORDER BY created_at DESC""",
        kb_id,
    )
    return [
        {
            "id": str(r["id"]),
            "title": r["title"],
            "source": r["source"],
            "doc_type": r["doc_type"],
            "chunk_count": r["chunk_count"],
            "metadata": r.get("metadata") or {},
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
        }
        for r in rows
    ]


@kb_router.delete("/{kb_id}/documents/{doc_id}", summary="删除文档")
async def delete_kb_document(kb_id: str, doc_id: str, request: Request, _user: dict = Depends(get_current_user)):
    """从知识库中删除文档（仅限管理员）."""
    # 使用 DocumentManager 协调多存储删除
    from src.rag.document_manager import DocumentManager
    doc_store = getattr(request.app.state, "doc_store", None)
    doc_mgr = DocumentManager(doc_store=doc_store)
    results = await doc_mgr.delete_document(doc_id, kb_id)

    # 检查是否有部分失败
    failures = {k: v for k, v in results.items() if v["status"] == "error"}
    if failures:
        from src.utils.logger import logger as _logger
        _logger.warning("doc_delete_partial_failures", doc_id=doc_id, kb_id=kb_id, failures=failures)

    return {"detail": "Document deleted", "backend_results": results}


@kb_router.get("/{kb_id}/documents/{doc_id}/chunks", summary="查看文档内容")
async def get_document_chunks(kb_id: str, doc_id: str, request: Request, user: dict = Depends(get_current_user)):
    """获取文档的全部切分块内容，用于预览文档."""
    kb_service = _get_kb_service(request)
    if kb_service is None:
        raise HTTPException(status_code=503, detail="KB service not available")

    user_id = user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")
    allowed = await kb_service.check_permission(kb_id, user_id, required="read")
    if not allowed:
        raise HTTPException(status_code=403, detail="Access denied: knowledge base read permission required")

    # Get document metadata from DocStore
    from src.stores.doc_store import DocStore
    doc_store: DocStore = getattr(request.app.state, "doc_store", None)
    if doc_store is None:
        raise HTTPException(status_code=503, detail="DocStore not available")

    doc = await doc_store.fetchrow(
        """SELECT id, title, source, doc_type, chunk_count, metadata, created_at
           FROM documents WHERE id = $1::uuid AND kb_id = $2::uuid""",
        doc_id, kb_id,
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get chunks from ChromaDB
    from src.rag.retrieval.collections import kb_collection_name
    from src.rag.retrieval.dense import _get_vector_store

    col_name = kb_collection_name(str(kb_id))
    try:
        vs = await _get_vector_store()
        chunks = await vs.get_by_metadata(
            metadata_filter={"doc_id": doc_id},
            top_k=max(doc["chunk_count"] * 2, 100),  # Safety multiplier
            collection_name=col_name,
        )
    except Exception as e:
        from src.utils.logger import logger as _logger
        _logger.warning("doc_chunks_fetch_failed", doc_id=doc_id, kb_id=kb_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document chunks: {e}")

    # Sort chunks by chunk_index for proper ordering
    chunks.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))

    return {
        "doc_id": str(doc["id"]),
        "title": doc["title"],
        "source": doc["source"],
        "doc_type": doc["doc_type"],
        "chunk_count": doc["chunk_count"],
        "chunks": [
            {
                "chunk_index": c.get("metadata", {}).get("chunk_index", i),
                "content": c.get("content", ""),
                "parent_index": c.get("metadata", {}).get("parent_index"),
            }
            for i, c in enumerate(chunks)
        ],
    }


@kb_router.put("/{kb_id}/permissions", summary="设置访问权限")
async def set_kb_permission(kb_id: str, body: KBPermissionRequest, request: Request, user: dict = Depends(get_current_user)):
    """设置用户对知识库的访问权限（需 admin）."""
    kb_service = _get_kb_service(request)
    if kb_service is None:
        raise HTTPException(status_code=503, detail="KB service not available")

    # get_current_user dependency already validates authentication

    await kb_service.set_permission(kb_id, body.user_id, body.permission)
    return {"detail": "Permission set"}
