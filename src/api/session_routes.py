"""Session and memory routes — extracted from routes.py for maintainability."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.dependencies import get_memory_service
from src.api.schemas import (
    CreateSessionRequest,
    MemoryItemResponse,
    SessionDetailResponse,
    SessionResponse,
)
from src.auth.dependencies import get_current_user
from src.config import settings

sessions_router = APIRouter(prefix="/sessions", tags=["sessions"])
memory_router = APIRouter(prefix="/memory", tags=["memory"])


# ── Auth helpers ───────────────────────────────────────────────


async def _get_user_id(request: Request) -> str:
    """Extract user_id from auth context.

    When auth is disabled, returns 'anonymous'.
    When auth is enabled, extracts the JWT Bearer token from the Authorization header,
    decodes it, and returns the user_id (sub claim). Raises 401 if token is missing or invalid.
    """
    if not settings.auth_enabled:
        return "anonymous"

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization token required")

    token = auth_header[7:]
    from src.auth.user_service import decode_jwt

    payload = decode_jwt(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return user_id


async def _verify_session_ownership(session_id: str, request: Request) -> dict:
    """Verify the session exists and (when auth enabled) belongs to the current user.

    Returns the session dict. Raises 404 if not found, 403 if owned by another user.
    """
    memory_service = getattr(request.app.state, "memory_service", None)
    if memory_service is None:
        raise HTTPException(status_code=503, detail="Memory service not available")

    if not settings.auth_enabled:
        session = await memory_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    # Auth enabled: verify ownership
    user_id = await _get_user_id(request)
    session = await memory_service.get_session(session_id, user_id=user_id)
    if not session:
        # Check if session exists at all (to distinguish 403 vs 404)
        any_session = await memory_service.get_session(session_id)
        if any_session:
            raise HTTPException(status_code=403, detail="Access denied: session belongs to another user")
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# ── Memory routes ──────────────────────────────────────────────


@memory_router.get("", response_model=list[MemoryItemResponse], summary="查看我的记忆列表")
async def list_my_memories(
    request: Request,
    session_id: str | None = None,
    current_user: dict = Depends(get_current_user),
    memory_service=Depends(get_memory_service),
):
    """返回当前用户的长期记忆。

    可选参数 session_id 用于按会话过滤记忆。
    不提供 session_id 时返回用户的所有记忆。
    """
    user_id = current_user.get("user_id") or "anonymous"

    memories = await memory_service.load_user_memories(user_id, session_id=session_id)
    return [MemoryItemResponse(**m) for m in memories]


@memory_router.delete("/{memory_id}", summary="删除某条记忆")
async def delete_memory(memory_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    """删除用户的一条长期记忆."""
    user_id = current_user.get("user_id") or "anonymous"

    memory_service = getattr(request.app.state, "memory_service", None)
    if memory_service is None:
        raise HTTPException(status_code=503, detail="Memory service not available")

    ok = await memory_service.delete_user_memory(memory_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"detail": "Memory deleted"}


@memory_router.delete("", summary="清除所有记忆")
async def clear_all_memories(request: Request, current_user: dict = Depends(get_current_user)):
    """清除当前用户的所有长期记忆."""
    user_id = current_user.get("user_id") or "anonymous"

    memory_service = getattr(request.app.state, "memory_service", None)
    if memory_service is None:
        raise HTTPException(status_code=503, detail="Memory service not available")

    count = await memory_service.clear_user_memories(user_id)
    return {"detail": f"Cleared {count} memories"}


# ── Session routes ─────────────────────────────────────────────


@sessions_router.post("", response_model=SessionResponse, status_code=201, summary="Create a new session")
async def create_session(body: CreateSessionRequest, request: Request):
    """Create a new conversation session."""
    memory_service = getattr(request.app.state, "memory_service", None)
    if memory_service is None:
        raise HTTPException(status_code=503, detail="Memory service not available")
    user_id = await _get_user_id(request)
    result = await memory_service.create_session(user_id=user_id, title=body.title)
    return SessionResponse(**result)


@sessions_router.get("", response_model=list[SessionResponse], summary="List sessions")
async def list_sessions(request: Request):
    """List all sessions for the current user, ordered by last activity.

    Sessions older than RAG_SESSION_TTL_DAYS (default 30) are automatically excluded.
    """
    memory_service = getattr(request.app.state, "memory_service", None)
    if memory_service is None:
        raise HTTPException(status_code=503, detail="Memory service not available")
    user_id = await _get_user_id(request)
    results = await memory_service.list_sessions(
        user_id=user_id,
        ttl_days=settings.session_ttl_days,
    )
    return [SessionResponse(**r) for r in results]


@sessions_router.get("/{session_id}", response_model=SessionDetailResponse, summary="Get session with history")
async def get_session_with_history(session_id: str, request: Request):
    """Get a single session with its full conversation history. Requires ownership."""
    memory_service = getattr(request.app.state, "memory_service", None)
    if memory_service is None:
        raise HTTPException(status_code=503, detail="Memory service not available")

    session = await _verify_session_ownership(session_id, request)
    history = await memory_service.load_history(session_id)
    return SessionDetailResponse(
        id=session["id"],
        user_id=session.get("user_id"),
        title=session.get("title"),
        summary=session.get("summary"),
        created_at=session.get("created_at"),
        updated_at=session.get("updated_at"),
        history=[{"role": h["role"], "content": h["content"]} for h in history],
    )


@sessions_router.delete("/{session_id}", summary="Delete a session")
async def delete_session(session_id: str, request: Request):
    """Delete a session and all its conversations. Requires ownership."""
    memory_service = getattr(request.app.state, "memory_service", None)
    if memory_service is None:
        raise HTTPException(status_code=503, detail="Memory service not available")

    # Verify ownership first
    await _verify_session_ownership(session_id, request)

    deleted = await memory_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"detail": "Session deleted"}
