"""LLM Call Tracker — Langfuse v2 generation span helper.

提供 track_llm_call() 辅助函数，从 OpenAI API 响应中提取 token 用量
并记录到 Langfuse generation span。使用 Langfuse v2 SDK generation() API。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.observability.client import is_langfuse_enabled, get_langfuse

logger = logging.getLogger(__name__)


def track_llm_call(
    name: str,
    model: str,
    start_time: float,
    response: Any | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """记录 LLM 调用到 Langfuse generation observation。

    使用 Langfuse v2 SDK generation() API 创建 generation 级 span，
    自动关联到当前 @observe() 上下文的 trace 和 parent observation。
    从 OpenAI 兼容 API 响应中提取 token 用量并记录。
    当 Langfuse 禁用时为零开销 no-op。

    Args:
        name: 调用名称（如 "intent_router_llm", "generate_llm"）。
        model: 模型名（如 "deepseek-chat"）。
        start_time: 调用开始时间（time.monotonic()）。
        response: OpenAI chat completion 响应对象（成功时传入）。
        error: 错误消息（失败时传入，与 response 互斥）。
        metadata: 额外的 metadata dict。
    """
    if not is_langfuse_enabled():
        return

    langfuse = get_langfuse()
    if langfuse is None:
        return

    latency_ms = (time.monotonic() - start_time) * 1000

    # Extract token usage from response
    usage: dict[str, int] = {}
    output_text: str | None = None
    if response is not None:
        try:
            if hasattr(response, "usage") and response.usage:
                usage["input"] = response.usage.prompt_tokens or 0
                usage["output"] = response.usage.completion_tokens or 0
                usage["total"] = response.usage.total_tokens or 0
            if hasattr(response, "choices") and response.choices:
                first = response.choices[0]
                if hasattr(first, "message") and first.message:
                    output_text = first.message.content[:500] if first.message.content else None
        except Exception:
            pass

    meta: dict[str, Any] = {
        "model": model,
        "latency_ms": round(latency_ms, 2),
        **(metadata or {}),
    }

    try:
        # Get current trace/observation context from @observe() decorator
        from langfuse.decorators import langfuse_context

        trace_id = langfuse_context.get_current_trace_id()
        parent_observation_id = langfuse_context.get_current_observation_id()

        # Langfuse v2 API: generation() creates a child generation under current span.
        # NOTE: usage_details parameter is broken in Langfuse v2.60.10 — data is silently
        # dropped. Use the deprecated `usage` parameter instead, which works correctly.
        gen = langfuse.generation(
            trace_id=trace_id,
            parent_observation_id=parent_observation_id,
            name=name,
            model=model,
            metadata=meta,
            usage=usage if usage else None,
            level="ERROR" if error else "DEFAULT",
            status_message=error if error else None,
            input=metadata.get("input") if metadata else None,
            output=output_text,
        )
        # End the generation to record latency
        gen.end()
    except Exception as exc:
        logger.debug("track_llm_call failed for %s: %s", name, exc)
