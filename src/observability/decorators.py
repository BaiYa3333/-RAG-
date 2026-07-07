"""可观测装饰器 — 项目级 Langfuse trace/span 封装.

提供 @trace_rag_node() 装饰器，基于 Langfuse v2 @observe() 为 RAG 管线节点
创建 span，统一 trace name、metadata 注入和错误处理。

当 RAG_LANGFUSE_ENABLED=false 时，装饰器为透传（零开销）。

Langfuse v2 SDK 使用 REST API 与 Langfuse Server 通信。
@observe() 装饰器自动管理 trace context 传播和 span 嵌套。
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable

from src.observability.client import is_langfuse_enabled

logger = logging.getLogger(__name__)


def trace_rag_node(
    name: str | None = None,
    as_type: str = "span",
):
    """为 RAG 节点创建 Langfuse observe span 的装饰器.

    当 Langfuse 禁用时，装饰器为透传（零开销）。
    使用 Langfuse v2 @observe() 实现，自动传播 trace context。

    Args:
        name: Span 名称，默认使用函数名。
        as_type: Langfuse observe type — "span" (默认) 或 "generation".
    """

    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        if not is_langfuse_enabled():
            return func

        try:
            from langfuse.decorators import observe

            # Langfuse v2 @observe() handles:
            # - trace context propagation via OpenTelemetry
            # - automatic parent/child span nesting
            # - latency measurement
            # - exception recording
            decorated = observe(
                name=func_name,
                as_type=as_type,
            )(func)

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await decorated(*args, **kwargs)
                except Exception:
                    raise

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return decorated(*args, **kwargs)
                except Exception:
                    raise

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        except ImportError:
            logger.warning("langfuse_observe_import_failed — decorator is a no-op")
            return func

    return decorator
