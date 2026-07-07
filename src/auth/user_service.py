"""用户认证服务 — 注册、登录、JWT 管理."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
from jose import jwt, JWTError

from src.config import settings
from src.stores.doc_store import DocStore
from src.utils.logger import logger


def hash_password(password: str) -> str:
    """使用 bcrypt 哈希密码."""
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码与哈希是否匹配."""
    password_bytes = plain_password.encode("utf-8")
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def create_jwt(user_id: str, role: str, expires_h: int | None = None) -> str:
    """创建 JWT access token."""
    expire_hours = expires_h or settings.jwt_expire_hours
    expire = datetime.now(timezone.utc) + timedelta(hours=expire_hours)
    payload = {
        "sub": user_id,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_jwt(token: str) -> dict | None:
    """解码 JWT token，返回 payload 或 None."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return payload
    except JWTError:
        return None


class UserService:
    """用户管理服务 — 封装 DocStore 操作."""

    def __init__(self, doc_store: DocStore):
        self._db = doc_store

    async def register(
        self,
        username: str,
        password: str,
        email: str | None = None,
        role: str = "user",
    ) -> dict:
        """注册新用户。返回用户信息 dict（不含密码哈希）."""
        password_hash = hash_password(password)

        # 检查用户名是否已存在
        existing = await self._db.fetchrow(
            "SELECT id FROM users_auth WHERE username = $1", username
        )
        if existing:
            raise ValueError(f"Username '{username}' already exists")

        if email:
            existing_email = await self._db.fetchrow(
                "SELECT id FROM users_auth WHERE email = $1", email
            )
            if existing_email:
                raise ValueError(f"Email '{email}' already registered")

        row = await self._db.fetchrow(
            """INSERT INTO users_auth (username, email, password_hash, role)
               VALUES ($1, $2, $3, $4)
               RETURNING id, username, email, role, is_active, created_at""",
            username, email, password_hash, role,
        )
        logger.info("user_registered", user_id=str(row["id"]), username=username)
        return {
            "id": str(row["id"]),
            "username": row["username"],
            "email": row.get("email"),
            "role": row["role"],
            "is_active": row["is_active"],
            "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        }

    async def login(self, username: str, password: str) -> dict | None:
        """验证用户名密码，成功返回 {access_token, token_type, user}."""
        row = await self._db.fetchrow(
            """SELECT id, username, email, password_hash, role, is_active
               FROM users_auth WHERE username = $1""",
            username,
        )
        if not row:
            return None

        if not row["is_active"]:
            logger.info("login_blocked_disabled", username=username)
            return None

        if not verify_password(password, row["password_hash"]):
            return None

        user_id = str(row["id"])
        role = row["role"]
        access_token = create_jwt(user_id, role)

        logger.info("user_login", user_id=user_id, username=username)
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user_id,
                "username": row["username"],
                "email": row.get("email"),
                "role": role,
                "is_active": row["is_active"],
            },
        }

    async def get_user_by_id(self, user_id: str) -> dict | None:
        """通过 ID 获取用户信息."""
        row = await self._db.fetchrow(
            """SELECT id, username, email, role, is_active, created_at, updated_at
               FROM users_auth WHERE id = $1::uuid""",
            user_id,
        )
        if not row:
            return None
        return {
            "id": str(row["id"]),
            "username": row["username"],
            "email": row.get("email"),
            "role": row["role"],
            "is_active": row["is_active"],
            "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
            "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        }

    async def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """修改密码。验证旧密码后更新为新密码."""
        row = await self._db.fetchrow(
            "SELECT password_hash FROM users_auth WHERE id = $1::uuid", user_id
        )
        if not row:
            return False
        if not verify_password(old_password, row["password_hash"]):
            return False

        new_hash = hash_password(new_password)
        await self._db.execute(
            "UPDATE users_auth SET password_hash = $1, updated_at = NOW() WHERE id = $2::uuid",
            new_hash, user_id,
        )
        logger.info("password_changed", user_id=user_id)
        return True

    async def deactivate_user(self, user_id: str) -> bool:
        """禁用用户."""
        result = await self._db.execute(
            "UPDATE users_auth SET is_active = FALSE, updated_at = NOW() WHERE id = $1::uuid",
            user_id,
        )
        affected = "UPDATE 1" in result if result else False
        if affected:
            logger.info("user_deactivated", user_id=user_id)
        return affected

    async def list_users(self) -> list[dict]:
        """列出所有用户（管理员功能）."""
        rows = await self._db.fetch(
            """SELECT id, username, email, role, is_active, created_at, updated_at
               FROM users_auth ORDER BY created_at DESC"""
        )
        return [
            {
                "id": str(r["id"]),
                "username": r["username"],
                "email": r.get("email"),
                "role": r["role"],
                "is_active": r["is_active"],
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
            }
            for r in rows
        ]
