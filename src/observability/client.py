"""Langfuse 客户端单例 — 生命周期管理.

提供 get_langfuse() 获取全局客户端实例，
flush_langfuse() 在应用关闭时确保数据发送。
通过 RAG_LANGFUSE_ENABLED 环境变量控制开启/关闭。

使用 Langfuse v2 SDK（REST API 通信，兼容 langfuse/langfuse:2 服务器）。
"""

import base64
import json
import logging
import os
import urllib.request
import urllib.error

from langfuse import Langfuse

from src.config import settings

logger = logging.getLogger(__name__)

_langfuse_client: Langfuse | None = None
_enabled: bool | None = None  # cached enabled flag

# ── Model pricing definitions ──────────────────────────────────────────
# Per-token prices in USD. Langfuse API expects price per unit (1 token).
# Update these when model pricing changes.
_MODEL_PRICES: list[dict] = [
    {
        "modelName": "deepseek-chat",
        "matchPattern": "(?i)^(deepseek-chat)$",
        "unit": "TOKENS",
        "inputPrice": 0.00000027,   # $0.27 / 1M input tokens
        "outputPrice": 0.00000110,  # $1.10 / 1M output tokens
        "tokenizerId": "openai",
    },
    {
        "modelName": "qwen-plus",
        "matchPattern": "(?i)^(qwen-plus)$",
        "unit": "TOKENS",
        "inputPrice": 0.0000008,    # $0.80 / 1M input tokens
        "outputPrice": 0.000002,    # $2.00 / 1M output tokens
        "tokenizerId": "openai",
    },
]


def is_langfuse_enabled() -> bool:
    """检查 Langfuse 监控是否启用.

    当 RAG_LANGFUSE_ENABLED=false 或密钥为占位值时返回 False.
    """
    global _enabled
    if _enabled is not None:
        return _enabled

    # Check the explicit toggle first
    enabled_str = getattr(settings, "langfuse_enabled", "true")
    if isinstance(enabled_str, bool):
        _enabled = enabled_str
    else:
        _enabled = str(enabled_str).strip().lower() in ("true", "1", "yes", "on")

    if not _enabled:
        logger.info("langfuse_disabled_by_config")
        return False

    # Check that keys are not placeholder values
    pk = settings.langfuse_public_key
    sk = settings.langfuse_secret_key
    if (not pk or pk.startswith("pk-lf-xxx")) or (not sk or sk.startswith("sk-lf-xxx")):
        logger.warning(
            "langfuse_disabled_placeholder_keys — keys are placeholder values; monitoring disabled."
        )
        _enabled = False
        return False

    return True


def _langfuse_api_request(method: str, path: str, body: dict | None = None) -> dict | None:
    """向 Langfuse Public API 发送 HTTP 请求 (stdlib urllib).

    Args:
        method: HTTP method (GET, POST).
        path: API path (e.g. "/api/public/models").
        body: Optional request body dict for POST.

    Returns:
        Parsed JSON response dict, or None on failure.
    """
    host = settings.langfuse_host.rstrip("/")
    url = f"{host}{path}"
    credentials = f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}"
    encoded = base64.b64encode(credentials.encode()).decode()

    data_bytes = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        logger.warning("langfuse_api_http_error method=%s url=%s status=%d", method, url, exc.code)
        return None
    except Exception as exc:
        logger.warning("langfuse_api_request_failed method=%s url=%s error=%s", method, url, exc)
        return None


def _ensure_langfuse_models() -> None:
    """确保 Langfuse 中已配置模型定价.

    Langfuse v2 自托管不会自动识别自定义模型的定价。
    此函数在客户端初始化后调用，通过 Langfuse Public API
    检查并创建模型定义（对已存在的模型跳过）。
    """
    try:
        # Fetch existing model names
        resp = _langfuse_api_request("GET", "/api/public/models")
        if resp is None:
            logger.warning("langfuse_model_sync_skip — cannot fetch existing models")
            return

        existing = {m["modelName"] for m in resp.get("data", [])}

        created = 0
        for model in _MODEL_PRICES:
            if model["modelName"] in existing:
                continue
            payload = dict(model)
            payload.setdefault("tokenizerConfig", {})
            result = _langfuse_api_request("POST", "/api/public/models", body=payload)
            if result is not None:
                logger.info(
                    "langfuse_model_created model=%s input_price=%s output_price=%s",
                    model["modelName"],
                    model["inputPrice"],
                    model["outputPrice"],
                )
                created += 1
            else:
                logger.warning(
                    "langfuse_model_create_failed model=%s",
                    model["modelName"],
                )

        if created:
            logger.info("langfuse_models_synced created=%d", created)
        else:
            logger.debug("langfuse_models_up_to_date")

    except Exception as exc:
        logger.warning("langfuse_model_sync_failed error=%s", exc)


def get_langfuse() -> Langfuse | None:
    """获取全局 Langfuse 客户端实例（单例).

    同时设置标准 LANGFUSE_* 环境变量以确保 @observe() 装饰器
    能通过 get_client() 找到配置。

    Returns:
        Langfuse 客户端实例，如果监控被禁用或初始化失败则返回 None.
    """
    global _langfuse_client

    if not is_langfuse_enabled():
        return None

    if _langfuse_client is None:
        try:
            # Set standard env vars for @observe() / langfuse_context compatibility
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
            os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)

            _langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("langfuse_client_initialized host=%s", settings.langfuse_host)

            # Sync model pricing definitions
            _ensure_langfuse_models()
        except Exception as exc:
            logger.warning("langfuse_init_failed error=%s", exc)
            return None

    return _langfuse_client


def flush_langfuse() -> None:
    """Flush 所有待发送的 trace/span 数据到 Langfuse Server.

    应在 FastAPI shutdown 时调用以确保数据不丢失.
    """
    global _langfuse_client
    if _langfuse_client is not None:
        try:
            _langfuse_client.flush()
            logger.info("langfuse_flushed")
        except Exception as exc:
            logger.warning("langfuse_flush_failed error=%s", exc)
