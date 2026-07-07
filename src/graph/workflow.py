"""LangGraph 1.0 工作流构建 — Quality-Gated 主干 + 局部 Agentic."""

import asyncio
import logging
import uuid
from typing import Literal

import time

from src.observability.langfuse_context import langfuse_context
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from src.graph.state import RAGState, initial_state
from src.graph.nodes.query_condenser import condense_query
from src.graph.nodes.intent_router import route_intent
from src.graph.nodes.retrieval import tier1_retrieve, tier2_retrieve
from src.graph.nodes.quality_gate import check_quality
from src.graph.nodes.rerank import rerank_docs
from src.graph.nodes.compress import compress_context
from src.graph.nodes.generate import generate_answer
from src.graph.nodes.fallback import safe_node
from src.graph.agent import run_analytical_agent
from src.observability.decorators import trace_rag_node
from src.observability.llm_tracker import track_llm_call

logger = logging.getLogger(__name__)

WORKFLOW_TIMEOUT_S = 60.0  # 最坏路径叠加 < 60s: condenser(2) + router(3) + tier1(4) + gate(1) + tier2(6) + gate(1) + rerank(8) + compress(4) + generate(15) = 44s


# ── 路由函数 ────────────────────────────────────────────────


def route_by_intent(
    state: RAGState,
) -> Literal["tier1_retrieve", "analytical_agent", "chitchat_answer"]:
    """Intent Router → 根据意图分流."""

    intent = state.get("intent", "factoid")
    if intent == "analytical":
        return "analytical_agent"
    if intent == "chitchat":
        return "chitchat_answer"
    # tabular 意图路由到标准 Tier1 检索路径（与 factoid 同款处理）：
    # 表格类问题通过 dense retrieval 召回，后续 quality_gate 评估质量
    if intent == "tabular":
        return "tier1_retrieve"
    # factoid / comparison / summary 等标准意图走 Tier1 检索
    return "tier1_retrieve"


def route_agent_result(
    state: RAGState,
) -> Literal["quality_gate", "tier2_retrieve"]:
    """Agent 后路由：fallback → 普通 complex 路径，成功 → quality_gate."""

    if state.get("agent_fallback", False):
        return "tier2_retrieve"
    return "quality_gate"


def route_after_rerank(
    state: RAGState,
) -> Literal["generate_answer", "compress_context"]:
    """Rerank 后路由：summary 和 analytical 意图走 compress 再生成，其余直接生成。"""
    intent = state.get("intent", "factoid")
    if intent in ("summary", "analytical"):  # 与 compress.py 节点判断对齐
        return "compress_context"
    return "generate_answer"


def route_by_gate(
    state: RAGState,
) -> Literal["generate_answer", "compress_context", "tier2_retrieve", "rerank_docs"]:
    """Quality Gate → gate_action 决定升级/跳过/生成.

    - pass: comparison/summary/analytical 意图走 rerank 后再生成（提升 context_precision）
    - pass + factoid: 直接生成（性能优先）
    - escalate: 升级到 Tier2
    - fallback_generate: 走 rerank → 生成
    """

    gate_log = state.get("gate_log") or []
    if not gate_log:
        return "generate_answer"

    # Filter by current retrieval_tier 避免并发节点写入导致的顺序依赖
    current_tier = state.get("retrieval_tier", 1)
    tier_logs = [g for g in gate_log if g.get("tier") == current_tier]
    last = tier_logs[-1] if tier_logs else gate_log[-1]
    action = last.get("action", "pass")
    tier = last.get("tier", 1)
    intent = state.get("intent", "factoid")

    if action == "pass":
        # Tier1 factoid: 低延迟直达生成；Tier2 factoid: 已升级过，先 rerank 压噪。
        if intent == "factoid":
            return "generate_answer" if tier == 1 else "rerank_docs"
        # comparison/summary/analytical/tabular → rerank 后再生成，提升排序精度
        if intent in ("comparison", "summary", "analytical", "tabular"):
            return "rerank_docs"
        return "generate_answer"

    elif action == "escalate":
        return "tier2_retrieve"

    else:  # fallback_generate
        # 所有意图都经过 rerank 后再生成
        return "rerank_docs"


# ── Fallback 包装节点 ───────────────────────────────────────


async def _noop_fallback(state: RAGState, exc: Exception) -> dict:
    return {"errors": [str(exc)]}


async def _chitchat_fallback(state: RAGState, exc: Exception) -> dict:
    return {"answer": "抱歉，我暂时无法回答这个问题。", "errors": [f"chitchat: {exc}"]}


@trace_rag_node(name="chitchat_answer")
async def chitchat_answer(state: RAGState) -> dict:
    """直接 LLM 回答闲聊/记忆类问题，不走检索流程。"""
    from src.llm.factory import create_llm
    query = state.get("query", "")
    chat_history = state.get("chat_history") or []
    try:
        llm = create_llm(model_name=state.get("model_name"))
        messages = [
            {"role": "system", "content": "你是一个友好的企业助手，直接、自然地回答用户的问题。"},
            *chat_history[-6:],
            {"role": "user", "content": query},
        ]
        llm_start = time.monotonic()
        resp = await llm.client.chat.completions.create(
            model=llm.model_id,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        track_llm_call(
            name="chitchat_answer_llm",
            model=llm.model_id,
            start_time=llm_start,
            response=resp,
        )
        answer = resp.choices[0].message.content.strip()
        langfuse_context.update_current_observation(
            node="chitchat_answer",
            query_len=len(query),
            answer_len=len(answer),
            skip_retrieval=True,
        )
        return {"answer": answer}
    except Exception as exc:
        langfuse_context.update_current_observation(
            node="chitchat_answer",
            query_len=len(query),
            error=str(exc),
            skip_retrieval=True,
        )
        return {"answer": "抱歉，我暂时无法回答这个问题。", "errors": [f"chitchat: {exc}"]}


async def _fallback_empty_docs(state: RAGState, exc: Exception) -> dict:
    return {"documents": [], "retrieval_scores": [], "errors": [str(exc)]}


async def _fallback_skip(state: RAGState, exc: Exception) -> dict:
    return {}


async def _fallback_agent(state: RAGState, exc: Exception) -> dict:
    return {
        "agent_fallback": True,
        "errors": [f"analytical_agent: {exc}"],
    }


# ── Wrapped nodes (safe_node) ───────────────────────────────


async def condenser_safe(state: RAGState) -> dict:
    return await safe_node(condense_query, _noop_fallback, state, timeout_s=2.0)


async def chitchat_safe(state: RAGState) -> dict:
    return await safe_node(chitchat_answer, _chitchat_fallback, state, timeout_s=5.0)


async def router_safe(state: RAGState) -> dict:
    return await safe_node(route_intent, _noop_fallback, state, timeout_s=3.0)


async def tier1_safe(state: RAGState) -> dict:
    return await safe_node(tier1_retrieve, _fallback_empty_docs, state, timeout_s=4.0)


async def tier2_safe(state: RAGState) -> dict:
    # Tier2: HyDE + embedding + BM25 + hybrid + RRF
    return await safe_node(tier2_retrieve, _fallback_empty_docs, state, timeout_s=6.0)


async def gate_safe(state: RAGState) -> dict:
    return await safe_node(check_quality, _noop_fallback, state, timeout_s=1.0)


async def rerank_safe(state: RAGState) -> dict:
    return await safe_node(rerank_docs, _fallback_skip, state, timeout_s=8.0)


async def compress_safe(state: RAGState) -> dict:
    return await safe_node(compress_context, _fallback_skip, state, timeout_s=4.0)


async def generate_safe(state: RAGState) -> dict:
    return await safe_node(generate_answer, _noop_fallback, state, timeout_s=15.0)


async def agent_safe(state: RAGState) -> dict:
    return await safe_node(run_analytical_agent, _fallback_agent, state, timeout_s=20.0)


# ── Workflow 构建 ───────────────────────────────────────────


def build_workflow() -> StateGraph:
    """构建 LangGraph 1.0 StateGraph。

    流程图:
        START → condenser → intent_router
                               │
            ┌──────────────────┼──────────────────┬──────────────────┐
            ▼                  ▼                  ▼                  ▼
        factoid /          analytical          chitchat           tabular
        comparison /                            │                  │
        summary              ▼                  ▼                  │
            │           agent_safe          chitchat_answer        │
            │            ┌──┴──┐                │                  │
            │         [ok]  [fallback]          ▼                  │
            │           │       │              END                 │
            │           ▼       │                                  │
            │      quality_gate│                                   │
            │           │      │                                   │
            └───────────┼──────┼───────────────────────────────────┘
                        ▼      ▼
                   tier1_retrieve
                        │
                   quality_gate
                    ┌───┴───┐
                [pass]   [escalate]    [fallback_generate]
                   │        │                │
              ┌────┴────┐   ▼                │
              │ intent? │ tier2_retrieve     │
              │         │   │                │
              │ factoid │ quality_gate       │
              │  │      │  ┌───┴───┐         │
              │  ▼      │[pass] [fail]       │
              │ gen.    │  │      │          │
              │         │  ▼      ▼          │
              │         │ rerank  rerank     │
              │         │  │      │          │
              └─┬─┬─────┘  │      │          │
                │ │   ┌────┴────┐ │          │
                │ │   │ intent? │ │          │
                │ │   │summary/ │ │          │
                │ │   │analytical│ │          │
                │ │   │compress │ │          │
                │ │   └────┬───┘ │          │
                │ │        ▼      │          │
                │ └──────→ generate ←────────┘
                │                   │
                └───────────────────┘
                                    ▼
                                   END

    Routing decisions:
    - intent_router: factoid/comparison/summary/tabular → tier1_retrieve;
      analytical → analytical_agent; chitchat → chitchat_answer
    - quality_gate: pass → generate/compress/rerank; escalate → tier2;
      fallback_generate → rerank → generate
    - rerank_docs: summary/analytical → compress_context → generate;
      others → generate directly
    - analytical_agent: success → quality_gate; fallback → tier2_retrieve
    """

    workflow = StateGraph(RAGState)

    # ── 节点挂载 ──
    workflow.add_node("query_condenser", condenser_safe)
    workflow.add_node("intent_router", router_safe)
    workflow.add_node("chitchat_answer", chitchat_safe)
    workflow.add_node("tier1_retrieve", tier1_safe)
    workflow.add_node("tier2_retrieve", tier2_safe)
    workflow.add_node("quality_gate", gate_safe)
    workflow.add_node("rerank_docs", rerank_safe)
    workflow.add_node("compress_context", compress_safe)
    workflow.add_node("generate_answer", generate_safe)
    workflow.add_node("analytical_agent", agent_safe)

    # ── 边 ──
    workflow.add_edge(START, "query_condenser")
    workflow.add_edge("query_condenser", "intent_router")

    # Intent Router → 4 路分流
    workflow.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "tier1_retrieve": "tier1_retrieve",
            "analytical_agent": "analytical_agent",
            "chitchat_answer": "chitchat_answer",
        },
    )

    # Tier1 → Quality Gate
    workflow.add_edge("tier1_retrieve", "quality_gate")

    # Quality Gate → pass / escalate
    workflow.add_conditional_edges(
        "quality_gate",
        route_by_gate,
        {
            "generate_answer": "generate_answer",
            "compress_context": "compress_context",
            "tier2_retrieve": "tier2_retrieve",
            "rerank_docs": "rerank_docs",
        },
    )

    # Compress → Generate
    workflow.add_edge("compress_context", "generate_answer")

    # Tier2 → Quality Gate (tier2_retrieve 内部已完成多路 RRF 融合)
    workflow.add_edge("tier2_retrieve", "quality_gate")

    # Rerank → Generate (or Compress → Generate for summary)
    workflow.add_conditional_edges(
        "rerank_docs",
        route_after_rerank,
        {
            "generate_answer": "generate_answer",
            "compress_context": "compress_context",
        },
    )

    # Chitchat → END (直接回答，不检索)
    workflow.add_edge("chitchat_answer", END)

    # Generate → END
    workflow.add_edge("generate_answer", END)

    # Agent → Gate (success) or Tier2 (fallback)
    workflow.add_conditional_edges(
        "analytical_agent",
        route_agent_result,
        {
            "quality_gate": "quality_gate",
            "tier2_retrieve": "tier2_retrieve",
        },
    )

    # Compile
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ── 便捷入口 ────────────────────────────────────────────────


async def run_query(
    query: str,
    chat_history: list[dict] | None = None,
    graph=None,
    model: str | None = None,
    session_id: str | None = None,
    language: str | None = None,
    user_id: str | None = None,
    kb_ids: list[str] | None = None,
) -> RAGState:
    """运行完整 RAG 工作流（非流式，含 60s 超时).

    Args:
        query: 用户查询
        chat_history: 可选的多轮对话历史
        graph: 可选，预编译的 LangGraph StateGraph。传入则复用，不传则 build_workflow()。
        model: 可选 LLM 模型名，传入后贯穿所有 LLM 节点。
        session_id: 可选持久会话 ID。
        language: 可选输出语言覆盖。
        user_id: 可选用户 ID。
        kb_ids: 可选目标知识库 ID 列表。
    """

    workflow = graph if graph is not None else build_workflow()
    state = initial_state(query, chat_history, model_name=model, session_id=session_id, language=language, user_id=user_id, kb_ids=kb_ids)
    config = {"configurable": {"thread_id": session_id or str(uuid.uuid4())}}

    try:
        result = await asyncio.wait_for(
            workflow.ainvoke(state, config=config),
            timeout=WORKFLOW_TIMEOUT_S,
        )
        return result
    except asyncio.TimeoutError:
        state["errors"].append("workflow: global timeout (60s)")
        state["answer"] = state.get("answer") or "请求处理超时，请稍后重试。"
        return state


async def run_query_stream(
    query: str,
    chat_history: list[dict] | None = None,
    graph=None,
    model: str | None = None,
    session_id: str | None = None,
    language: str | None = None,
    user_id: str | None = None,
    kb_ids: list[str] | None = None,
):
    """运行完整 RAG 工作流（流式).

    Args:
        query: 用户查询
        chat_history: 可选的多轮对话历史
        graph: 可选，预编译的 LangGraph StateGraph。传入则复用，不传则 build_workflow()。
        model: 可选 LLM 模型名，传入后贯穿所有 LLM 节点。
        session_id: 可选持久会话 ID。
        language: 可选输出语言覆盖。
        user_id: 可选用户 ID。
        kb_ids: 可选目标知识库 ID 列表。
    """

    workflow = graph if graph is not None else build_workflow()
    state = initial_state(query, chat_history, model_name=model, session_id=session_id, language=language, user_id=user_id, kb_ids=kb_ids, stream_mode=True)
    config = {"configurable": {"thread_id": session_id or str(uuid.uuid4())}}

    try:
        async for chunk in workflow.astream(state, config=config):
            yield chunk
    except asyncio.TimeoutError:
        yield {"error": "workflow timeout (60s)"}
