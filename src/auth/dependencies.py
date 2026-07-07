"""FastAPI dependencies for authentication — JWT + API Key 双模式."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.config import settings

security = HTTPBearer(auto_error=False)


def get_auth_service(request: Request):
    """Dependency: get the AuthService from app state."""
    return request.app.state.auth_service


def get_user_service(request: Request):
    """Dependency: get the UserService from app state."""
    return getattr(request.app.state, "user_service", None)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """Extract and validate the current user from Bearer token.

    认证优先级: JWT > API Key > anonymous
    - 先尝试 JWT 解码（用户登录）
    - 失败则尝试 API Key 验证（向后兼容）
    - auth_enabled=False 时返回 anonymous
    """
    if credentials is None:
        if not settings.auth_enabled:
            return {"user_id": "anonymous", "role": "user", "api_key_id": None}
        raise HTTPException(status_code=401, detail="Authorization token required")

    token = credentials.credentials

    # 优先尝试 JWT 解析（用户登录）
    from src.auth.user_service import decode_jwt
    payload = decode_jwt(token)
    if payload:
        user_id = payload.get("sub")
        role = payload.get("role", "user")
        if user_id:
            # 检查用户是否仍处于激活状态
            user_service = getattr(request.app.state, "user_service", None)
            if user_service:
                user = await user_service.get_user_by_id(user_id)
                if user and user.get("is_active"):
                    return {"user_id": user_id, "role": role, "auth_method": "jwt"}
                elif user:
                    raise HTTPException(status_code=403, detail="Account disabled")
            # 即使 user_service 不可用，仍接受有效的 JWT
            return {"user_id": user_id, "role": role, "auth_method": "jwt"}

    # 降级到 API Key 验证（向后兼容）
    auth_service = getattr(request.app.state, "auth_service", None)
    if auth_service:
        user = await auth_service.validate_api_key(token)
        if user:
            return {**user, "auth_method": "api_key"}

    raise HTTPException(status_code=401, detail="Invalid credentials")


