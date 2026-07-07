"""Query Condenser 节点 — 多轮对话指代消解 + 用户记忆注入."""

import logging
import time

from src.observability.langfuse_context import langfuse_context

from src.llm.factory import create_llm
from src.graph.state import RAGState
from src.observability.decorators import trace_rag_node
from src.observability.llm_tracker import track_llm_call

logger = logging.getLogger(__name__)

# 模块级 memory_service 引用，由 app startup 注入
_memory_service = None


def set_memory_service(service) -> None:
    global _memory_service
    _memory_service = service


CONDENSE_SYSTEM = (
    "你是一个查询改写助手。基于对话历史和用户信息，将用户的当前问题改写为一个独立的、"
    "无需上下文即可理解的查询。如果当前问题已经独立完整，直接返回原问题。"
    "只返回改写后的查询文本，不要加任何额外说明。"
)


@trace_rag_node(name="query_condenser")
async def condense_query(state: RAGState) -> dict:
    """多轮指代消解：chat_history 非空时用 LLM 将当前 query 压缩为独立查询。

    同时注入用户长期记忆到 chat_history 中。
    返回部分 state 更新（query 字段，可能注入 memory_context）。
    """

    query = state.get("query", "")
    chat_history = state.get("chat_history") or []
    user_id = state.get("user_id", "")
    session_id = state.get("session_id")

    result: dict = {}
    memory_injected = False
    coref_resolved = False

    # ── 注入用户长期记忆（按会话隔离） ──
    if user_id and _memory_service:
        try:
            memories = await _memory_service.load_user_memories(
                user_id, limit=10, session_id=session_id,
            )
            if memories:
                memory_injected = True
                memory_context = "\n".join(
                    f"- [{m.get('memory_type', 'fact')}] {m.get('content', '')}"
                    for m in memories
                )
                # 将记忆作为 system 消息预置到 chat_history
                memory_msg = {
                    "role": "system",
                    "content": f"[关于用户的已知信息]\n{memory_context}",
                }
                # 注入到 chat_history 开头（持久化用，不修改 state）
                chat_history = [memory_msg] + chat_history
                result["memory_context"] = memory_context
                logger.debug(
                    "injected_user_memories",
                    user_id=user_id,
                    session_id=session_id,
                    count=len(memories),
                )
        except Exception as exc:
            logger.warning("memory_inject_failed: %s", exc)

    if not chat_history:
        langfuse_context.update_current_observation(
            node="query_condenser",
            query_len=len(query),
            memory_injected=memory_injected,
            coref_resolved=False,
        )
        return result

    # 过滤 role=system 消息（session summary），仅保留 user/assistant 轮次用于指代消解
    user_assistant_turns = [m for m in chat_history if m.get("role") != "system"]
    if not user_assistant_turns:
        langfuse_context.update_current_observation(
            node="query_condenser",
            query_len=len(query),
            memory_injected=memory_injected,
            coref_resolved=False,
        )
        return result

    try:
        llm = create_llm(model_name=state.get("model_name"))
        history_text = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in user_assistant_turns[-6:]  # 最近 6 轮 user/assistant
        )

        llm_start = time.monotonic()
        resp = await llm.client.chat.completions.create(
            model=llm.model_id,
            messages=[
                {"role": "system", "content": CONDENSE_SYSTEM},
                {"role": "user", "content": (
                    f"对话历史:\n{history_text}\n\n"
                    f"当前问题: {query}\n\n"
                    f"独立查询:"
                )},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        track_llm_call(
            name="query_condenser_llm",
            model=llm.model_id,
            start_time=llm_start,
            response=resp,
        )
        condensed = resp.choices[0].message.content.strip()
        if condensed and condensed != query:
            result["query"] = condensed
            coref_resolved = True

        langfuse_context.update_current_observation(
            node="query_condenser",
            query_len=len(query),
            memory_injected=memory_injected,
            coref_resolved=coref_resolved,
            condensed_query_len=len(condensed) if condensed else 0,
        )
        return result
    except Exception:
        langfuse_context.update_current_observation(
            node="query_condenser",
            query_len=len(query),
            memory_injected=memory_injected,
            coref_resolved=False,
        )
        return result
