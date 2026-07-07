"""Langfuse v2 上下文助手 — 设置 span/trace 属性.

Langfuse v2 SDK 原生提供 langfuse_context API。
当前 span 的 metadata 通过 langfuse_context.update_current_observation() 设置。
trace 级 metadata 通过 langfuse_context.update_current_trace() 设置。

提供辅助函数:
    update_current_observation(**kwargs) — 更新当前 observation (span) metadata
    update_current_trace(**kwargs) — 更新 trace 级 metadata
"""

from __future__ import annotations

import logging
from typing import Any

from src.observability.client import is_langfuse_enabled

logger = logging.getLogger(__name__)


def _flatten_attributes(attributes: dict[str, Any]) -> dict[str, str | int | float | bool] | None:
    """将属性 dict 扁平化为 Langfuse v2 metadata 兼容的类型，跳过 None 值."""
    if not attributes:
        return None
    flat: dict[str, str | int | float | bool] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif isinstance(value, (list, dict)):
            import json

            flat[key] = json.dumps(value, ensure_ascii=False, default=str)
        else:
            flat[key] = str(value)
    return flat if flat else None


def update_current_observation(**attributes: Any) -> None:
    """更新当前 Langfuse observation (span) 的 metadata 属性.

    通过 Langfuse v2 SDK 原生 langfuse_context 设置。
    当 Langfuse 禁用或不在 observe 上下文中时，静默跳过。

    Usage:
        update_current_observation(node="intent_router", intent="factoid", confidence=0.95)
    """
    if not is_langfuse_enabled():
        return

    try:
        from langfuse.decorators import langfuse_context

        flat = _flatten_attributes(attributes)
        if flat:
            langfuse_context.update_current_observation(metadata=flat)
    except Exception:
        pass  # Fail silently — observability must not break business logic


def update_current_trace(**attributes: Any) -> None:
    """更新当前 Langfuse trace 的 metadata 属性.

    通过 Langfuse v2 SDK 原生 langfuse_context 设置。

    Usage:
        update_current_trace(session_id="...", user_id="...", query="...", model="deepseek-chat")
    """
    if not is_langfuse_enabled():
        return

    try:
        from langfuse.decorators import langfuse_context

        flat = _flatten_attributes(attributes)
        if flat:
            langfuse_context.update_current_trace(metadata=flat)
    except Exception:
        pass  # Fail silently — observability must not break business logic
