"""SSE streaming helpers — typed event emitters for streaming endpoints."""

from __future__ import annotations

import json
from typing import AsyncGenerator

from src.api.sse_events import SSEEventType


async def sse_event(event_type: SSEEventType | str, payload: dict) -> str:
    """Format a single SSE event string."""
    event_name = event_type.value if isinstance(event_type, SSEEventType) else event_type
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"


async def emit_event(event_type: SSEEventType | str, payload: dict):
    """Yield a formatted SSE event. Used inside async generators."""
    return await sse_event(event_type, payload)


async def wrap_stream_with_events(
    stream: AsyncGenerator,
    model: str = "",
    session_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Wrap a raw LangGraph astream generator with typed SSE events.

    This is the primary helper used by /chat/stream.
    Accumulates token chunks and includes the full answer in the done event.
    """
    last_event_type: str | None = None
    # 累积最终状态元信息，用于 done 事件
    final_intent: str = ""
    final_retrieval_tier: int = 1
    final_documents_count: int = 0
    final_answer: str = ""

    async for chunk in stream:
        if not isinstance(chunk, dict):
            continue

        # Handle error events directly
        if "error" in chunk:
            yield await sse_event(SSEEventType.error, {"node": "", "message": chunk["error"]})
            continue

        # Process graph node outputs
        for node_name, node_state in chunk.items():
            # 从节点输出捕获最终状态元信息
            if node_name == "intent_router":
                final_intent = node_state.get("intent", final_intent)
            elif node_name in ("tier1_retrieve", "tier2_retrieve"):
                final_retrieval_tier = 2 if node_name == "tier2_retrieve" else 1
                docs = node_state.get("documents") or []
                final_documents_count = len(docs)
            elif node_name in ("generate_answer", "chitchat_answer"):
                answer = node_state.get("answer", "")
                if answer:
                    final_answer += answer
                docs = node_state.get("documents") or []
                if not final_documents_count and docs:
                    final_documents_count = len(docs)
            elif node_name == "analytical_agent":
                docs = node_state.get("documents") or []
                if docs:
                    final_documents_count = len(docs)

            event_str, event_type = _node_to_sse(node_name, node_state)
            if event_str and event_type != last_event_type:
                last_event_type = event_type
                yield event_str

    # Emit done event with accumulated metadata including full answer
    done_payload = {
        "answer": final_answer,
        "status": "complete",
        "intent": final_intent,
        "retrieval_tier": final_retrieval_tier,
        "documents_count": final_documents_count,
        "model": model,
        "session_id": session_id,
    }
    yield await sse_event(SSEEventType.done, done_payload)


def _node_to_sse(node_name: str, state: dict) -> tuple[str | None, str | None]:
    """Convert a LangGraph node output chunk to an SSE event string and type."""
    if node_name == "intent_router":
        payload = {
            "intent": state.get("intent", ""),
            "confidence": state.get("router_confidence", 0.0),
        }
        return (
            f"event: intent\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n",
            "intent",
        )

    if node_name in ("tier1_retrieve", "tier2_retrieve"):
        docs = state.get("documents") or []
        tier = 2 if node_name == "tier2_retrieve" else 1
        payload = {
            "tier": tier,
            "doc_count": len(docs),
            "top_scores": [d.get("score", d.get("rrf_score", 0.0)) for d in docs[:5]],
        }
        return (
            f"event: retrieval_chunk\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n",
            "retrieval_chunk",
        )

    if node_name == "generate_answer":
        answer = state.get("answer", "")
        if answer:
            return (
                f"event: token\ndata: {json.dumps({'token': answer}, ensure_ascii=False)}\n\n",
                "token",
            )
        return None, None

    if node_name == "quality_gate":
        gate_log = state.get("gate_log") or []
        last = gate_log[-1] if gate_log else {}
        payload = {
            "action": last.get("action", ""),
            "tier": last.get("tier", 0),
            "avg_score": last.get("avg_score"),
            "reason": last.get("reason"),
        }
        return (
            f"event: quality_gate\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n",
            "quality_gate",
        )

    return None, None
