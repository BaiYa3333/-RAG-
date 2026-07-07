"""Langfuse v2 兼容代理 — 提供统一的 langfuse_context API。

Langfuse v2 SDK 原生提供 langfuse.decorators.langfuse_context。
本模块封装一层代理对象，暴露与原生相同的
update_current_observation() 和 update_current_trace() 方法。

内部委托给 src.observability.context 模块实现。

Usage:
    from src.observability.langfuse_context import langfuse_context
    langfuse_context.update_current_observation(node="intent_router", intent="factoid")
    langfuse_context.update_current_trace(session_id="...", user_id="...")
"""

from __future__ import annotations

import logging
from typing import Any

from src.observability.context import update_current_observation, update_current_trace

logger = logging.getLogger(__name__)


class _LangfuseContext:
    """Langfuse v2 langfuse_context 代理对象.

    当 Langfuse 禁用时，所有方法为 no-op（零开销）。
    当启用时，通过 langfuse_context 原生 API 设置 span/trace 属性。
    """

    @staticmethod
    def update_current_observation(**attributes: Any) -> None:
        """更新当前 Langfuse span 的 metadata 属性（兼容 v2 API)."""
        update_current_observation(**attributes)

    @staticmethod
    def update_current_trace(**attributes: Any) -> None:
        """更新当前 Langfuse trace 的 metadata 属性（兼容 v2 API)."""
        update_current_trace(**attributes)


# 模块级单例，兼容 v2 的 `from langfuse.decorators import langfuse_context` 用法
langfuse_context = _LangfuseContext()
