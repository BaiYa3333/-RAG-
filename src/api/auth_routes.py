"""用户认证路由 — 注册 / 登录 / 个人信息."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.auth.dependencies import get_current_user, get_user_service
from src.auth.user_service import UserService, decode_jwt
from src.config import settings

auth_router = APIRouter(prefix="/auth", tags=["auth"])


async def _get_user_id_from_request(request: Request) -> str | None:
    """Extract user_id from request Authorization header, or None."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    payload = decode_jwt(token)
    if not payload:
        # Fallback: API key
        auth_service = getattr(request.app.state, "auth_service", None)
        if auth_service:
            user = await auth_service.validate_api_key(token)
            if user:
                return user.get("user_id")
        return None
    return payload.get("sub")


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, description="用户名")
    password: str = Field(..., min_length=8, max_length=128, description="密码（最少8位）")
    email: str | None = Field(default=None, max_length=255, description="邮箱（可选）")


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, description="用户名")
    password: str = Field(..., min_length=1, description="密码")


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: str
    username: str
    email: str | None = None
    role: str
    is_active: bool
    created_at: str | None = None


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


@auth_router.post("/register", response_model=UserResponse, status_code=201, summary="用户注册")
async def register(body: RegisterRequest, request: Request):
    """注册新用户。当 RAG_REGISTRATION_ENABLED=false 时仅管理员可调用."""
    user_service: UserService = get_user_service(request)
    if user_service is None:
        raise HTTPException(status_code=503, detail="User service not available")

    # 如果关闭开放注册，需要已登录用户
    if not settings.registration_enabled:
        user_id = await _get_user_id_from_request(request)
        if not user_id:
            raise HTTPException(status_code=403, detail="Registration is disabled")

    try:
        user = await user_service.register(
            username=body.username,
            password=body.password,
            email=body.email,
            role="user",
        )
        return UserResponse(**user)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@auth_router.post("/login", response_model=LoginResponse, summary="用户登录")
async def login(body: LoginRequest, request: Request):
    """使用用户名+密码登录，返回 JWT access_token 并设置 HttpOnly Cookie."""
    user_service: UserService = get_user_service(request)
    if user_service is None:
        raise HTTPException(status_code=503, detail="User service not available")

    result = await user_service.login(body.username, body.password)
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    response = JSONResponse(content=result)
    # Set JWT cookie for server-side page auth
    response.set_cookie(
        key="rag_jwt",
        value=result["access_token"],
        max_age=settings.jwt_expire_hours * 3600,
        httponly=True,
        samesite="lax",
        secure=False,  # Set True in production with HTTPS
        path="/",
    )
    return response


@auth_router.post("/logout", summary="用户登出")
async def logout():
    """清除 JWT Cookie，客户端也应删除 token."""
    response = JSONResponse(content={"detail": "Logged out (client should discard token)"})
    response.delete_cookie("rag_jwt", path="/")
    return response


@auth_router.get("/me", response_model=UserResponse, summary="获取当前用户信息")
async def get_me(request: Request, user: dict = Depends(get_current_user)):
    """返回当前认证用户的详细信息."""
    user_service: UserService = get_user_service(request)
    user_id = user.get("user_id")

    if user_service and user_id and user_id != "anonymous":
        user_info = await user_service.get_user_by_id(user_id)
        if user_info:
            return UserResponse(**user_info)

    # Fallback: 返回 basic auth 信息
    return UserResponse(
        id=user_id or "unknown",
        username=user_id or "anonymous",
        role=user.get("role", "user"),
        is_active=True,
    )


@auth_router.put("/me/password", summary="修改密码")
async def change_password(
    body: ChangePasswordRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """修改当前用户密码（需提供旧密码）."""
    user_service: UserService = get_user_service(request)
    if user_service is None:
        raise HTTPException(status_code=503, detail="User service not available")

    user_id = user.get("user_id")
    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Login required")

    success = await user_service.change_password(user_id, body.old_password, body.new_password)
    if not success:
        raise HTTPException(status_code=400, detail="Old password is incorrect")

    return {"detail": "Password changed successfully"}


@auth_router.get("/users", response_model=list[UserResponse], summary="列出所有用户")
async def list_users(
    request: Request,
    user: dict = Depends(get_current_user),
):
    """列出所有用户."""
    user_service: UserService = get_user_service(request)
    if user_service is None:
        raise HTTPException(status_code=503, detail="User service not available")

    users = await user_service.list_users()
    return [UserResponse(**u) for u in users]
