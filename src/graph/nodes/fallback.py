"""统一 fallback 工具 — safe_node 包装器 + 超时控制."""

import asyncio
import logging
from typing import Any, Callable, Coroutine

from src.graph.state import RAGState

logger = logging.getLogger(__name__)

DEFAULT_NODE_TIMEOUT_S = 5.0


async def safe_node(
    node_fn: Callable[[RAGState], Coroutine[Any, Any, dict]],
    fallback_fn: Callable[[RAGState, Exception], Coroutine[Any, Any, dict]],
    state: RAGState,
    timeout_s: float = DEFAULT_NODE_TIMEOUT_S,
) -> dict:
    """包装节点执行：try/except + asyncio.wait_for + 错误日志。

    Args:
        node_fn: 正常节点执行函数 (state) → partial_state_dict
        fallback_fn: 降级函数 (state, exception) → partial_state_dict
        state: 当前工作流状态
        timeout_s: 节点超时秒数 (默认 5s)

    Returns:
        部分 state 更新 dict
    """

    node_name = getattr(node_fn, "__name__", "unknown_node")

    try:
        return await asyncio.wait_for(node_fn(state), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning("[safe_node] %s 超时 (%.1fs) → fallback", node_name, timeout_s)
        exc = TimeoutError(f"{node_name} timed out after {timeout_s}s")
        result = await fallback_fn(state, exc)
        result.setdefault("errors", []).append(f"{node_name}: timeout")
        return result
    except Exception as exc:
        logger.warning("[safe_node] %s 异常: %s → fallback", node_name, exc)
        result = await fallback_fn(state, exc)
        result.setdefault("errors", []).append(f"{node_name}: {exc}")
        return result
