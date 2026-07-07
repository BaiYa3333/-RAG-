"""Admin API endpoints — API key management and usage analytics."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.auth.dependencies import get_auth_service, get_current_user
from src.auth.service import AuthService

admin_router = APIRouter(prefix="/admin", tags=["admin"])


class CreateApiKeyRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Associated user ID")
    role: str = Field(default="user", pattern="^(admin|user|viewer)$")
    label: str | None = Field(default=None)


class ApiKeyResponse(BaseModel):
    id: str
    user_id: str
    role: str
    label: str | None = None
    revoked: bool = False
    created_at: str | None = None
    last_used: str | None = None


class CreateApiKeyResponse(BaseModel):
    id: str
    raw_key: str
    user_id: str
    role: str
    label: str | None = None
    message: str = "Store this key securely — it will not be shown again."


class UsageResponse(BaseModel):
    total_requests: int
    total_tokens: int
    avg_latency_ms: float
    total_cost: float


# ── API Keys ──────────────────────────────────────────────────


@admin_router.post("/api-keys", response_model=CreateApiKeyResponse, summary="Create a new API key")
async def create_api_key(
    body: CreateApiKeyRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
):
    """Admin creates a new API key. The raw key is returned once."""
    auth_service: AuthService = get_auth_service(request)
    raw_key, key_id = await auth_service.create_api_key(
        user_id=body.user_id, role=body.role, label=body.label,
    )
    return CreateApiKeyResponse(
        id=key_id, raw_key=raw_key, user_id=body.user_id, role=body.role, label=body.label,
    )


@admin_router.get("/api-keys", response_model=list[ApiKeyResponse], summary="List all API keys")
async def list_api_keys(
    request: Request,
    _user: dict = Depends(get_current_user),
):
    """Admin lists all API keys (key values are masked)."""
    auth_service: AuthService = get_auth_service(request)
    return await auth_service.list_api_keys()


@admin_router.delete("/api-keys/{key_id}", summary="Revoke an API key")
async def revoke_api_key(
    key_id: str,
    request: Request,
    _user: dict = Depends(get_current_user),
):
    """Admin revokes an API key."""
    auth_service: AuthService = get_auth_service(request)
    revoked = await auth_service.revoke_api_key(key_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"detail": "API key revoked"}


# ── Usage ─────────────────────────────────────────────────────


@admin_router.get("/usage", response_model=UsageResponse, summary="Get usage analytics")
async def get_usage(
    request: Request,
    user_id: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    _user: dict = Depends(get_current_user),
):
    """Admin queries aggregated usage stats with optional filters."""
    auth_service: AuthService = get_auth_service(request)
    return await auth_service.get_usage(user_id=user_id, from_date=from_date, to_date=to_date)
