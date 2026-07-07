"""Intent Router 节点 — 4 路意图分类 + 检索策略建议."""

import json
import logging

import time

from src.observability.langfuse_context import langfuse_context

from src.llm.factory import create_llm
from src.graph.state import RAGState
from src.observability.decorators import trace_rag_node
from src.observability.llm_tracker import track_llm_call

logger = logging.getLogger(__name__)

ROUTER_SYSTEM = (
    "你是一个查询意图分类器。分析用户查询并返回 JSON。\n\n"
    "意图类型:\n"
    "- chitchat: 闲聊、问候、感谢、记忆类（记住名字/你好/谢谢/你是谁）→ 不需要检索\n"
    "- factoid: 简单事实查询 (是什么/谁/何时), 短答案即可\n"
    "- comparison: 对比类查询 (对比/区别/vs/优劣)\n"
    "- summary: 总结类查询 (总结/概括/要点/概述)\n"
    "- analytical: 分析类查询, 需要拆解成多个子问题 (多跳推理/原因分析/趋势)\n"
    "- tabular: 需要表格输出的查询（对比多个选项/列举数据/价格表/功能矩阵）→ suggested_k=10\n\n"
    "输出 JSON 格式:\n"
    '{"intent": "<intent>", "suggested_k": <0|5|10|20>, '
    '"suggested_tools": ["<tool>", ...], "confidence": <0.0-1.0>}\n\n'
    "suggested_k: chitchat → 0, factoid → 5, comparison → 10, summary → 20, analytical → 0, tabular → 10\n"
    "只返回 JSON，不要加其他文本。"
)

# 默认 fallback 值
FALLBACK_INTENT = {
    "intent": "factoid",
    "suggested_k": 5,
    "suggested_tools": ["dense_search"],
    "confidence": 0.0,
}


@trace_rag_node(name="intent_router")
async def route_intent(state: RAGState) -> dict:
    """LLM 判别查询意图 + 输出检索策略建议。

    返回部分 state 更新（intent, suggested_k, suggested_tools, router_confidence）。
    """

    query = state.get("query", "")

    llm_start = time.monotonic()
    model_id = state.get("model_name") or "unknown"
    try:
        llm = create_llm(model_name=state.get("model_name"))
        model_id = llm.model_id
        resp = await llm.client.chat.completions.create(
            model=llm.model_id,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        track_llm_call(
            name="intent_router_llm",
            model=llm.model_id,
            start_time=llm_start,
            response=resp,
        )
        raw = resp.choices[0].message.content.strip()
        result = json.loads(raw)

        intent = result.get("intent", "factoid")
        if intent not in ("factoid", "comparison", "summary", "analytical", "chitchat", "tabular"):
            intent = "factoid"

        langfuse_context.update_current_observation(
            node="intent_router",
            intent=intent,
            confidence=result.get("confidence", 0.5),
            suggested_k=result.get("suggested_k", 5),
            suggested_tools=result.get("suggested_tools", ["dense_search"]),
        )
        return {
            "intent": intent,
            "suggested_k": result.get("suggested_k", 5),
            "suggested_tools": result.get("suggested_tools", ["dense_search"]),
            "router_confidence": result.get("confidence", 0.5),
        }
    except Exception as exc:
        logger.warning("[intent_router] 分类失败: %s → fallback factoid", exc)
        track_llm_call(
            name="intent_router_llm",
            model=model_id,
            start_time=llm_start,
            error=str(exc),
        )
        langfuse_context.update_current_observation(
            node="intent_router",
            intent="factoid",
            confidence=0.0,
            error=str(exc),
        )
        return {**FALLBACK_INTENT, "errors": [f"intent_router: {exc}"]}
