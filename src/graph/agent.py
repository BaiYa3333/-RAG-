"""局部 Agent 子图 — 仅 analytical 意图激活 (LangGraph 1.0)."""

import asyncio
import logging
import time
from typing import Any

from src.observability.langfuse_context import langfuse_context

from src.llm.factory import create_llm
from src.rag.retrieval.dense import dense_search
from src.rag.embeddings.text_embedding_v4 import TextEmbeddingV4
from src.observability.decorators import trace_rag_node
from src.observability.llm_tracker import track_llm_call

logger = logging.getLogger(__name__)

# 模块级单例，避免每次 tool 调用都新建 HTTP 连接
_agent_embedder: "TextEmbeddingV4 | None" = None
_agent_llms: dict[str, Any] = {}


def _get_agent_embedder():
    global _agent_embedder
    if _agent_embedder is None:
        _agent_embedder = TextEmbeddingV4()
    return _agent_embedder


def _get_agent_llm(model_name: str | None = None):
    key = model_name or "__default__"
    if key not in _agent_llms:
        _agent_llms[key] = create_llm(model_name=model_name, temperature=0.0)
    return _agent_llms[key]


AGENT_MAX_ITER = 5
AGENT_TIMEOUT_S = 20.0

DECOMPOSE_SYSTEM = (
    "你是一个查询分析专家。将复杂查询拆解为独立的子问题列表。\n"
    "每个子问题应该是独立可检索的。最多拆解 4 个子问题。\n"
    "返回 JSON 格式: {\"sub_queries\": [\"子问题1\", \"子问题2\", ...]}"
)

AGGREGATE_SYSTEM = (
    "你是一个信息汇总专家。基于多个子问题的检索结果，"
    "为原始复杂查询生成一个结构化的中间答案。"
    "包含关键发现和子问题之间的关联。"
)

# ── Agent Tools ──────────────────────────────────────────────


async def _decompose_query_tool(query: str, model_name: str | None = None) -> dict:
    """Tool: 拆解复杂查询为子问题列表."""
    try:
        llm = _get_agent_llm(model_name)
        llm_start = time.monotonic()
        resp = await llm.client.chat.completions.create(
            model=llm.model_id,
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM},
                {"role": "user", "content": f"拆解以下查询:\n{query}"},
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        track_llm_call(
            name="agent_decompose_llm",
            model=llm.model_id,
            start_time=llm_start,
            response=resp,
        )

        import json
        data = json.loads(resp.choices[0].message.content.strip())
        sub_queries = data.get("sub_queries", [query])[:4]

        return {
            "sub_queries": sub_queries,
            "count": len(sub_queries),
        }
    except Exception as exc:
        logger.warning("[agent.decompose] 拆解失败: %s", exc)
        return {"sub_queries": [query], "count": 1, "error": str(exc)}


async def _search_docs_tool(sub_query: str, k: int = 5) -> dict:
    """Tool: 对子问题执行 dense search（复用持久连接）."""
    try:
        from src.rag.retrieval.dense import _get_vector_store

        embedder = _get_agent_embedder()
        embeddings = await embedder.embed([sub_query])
        if not embeddings:
            return {"query": sub_query, "results": [], "count": 0, "error": "embedding failed"}

        vs = await _get_vector_store()
        results = await dense_search(embeddings[0], top_k=k, vector_store=vs)
        return {
            "query": sub_query,
            "results": [
                {"content": r.get("content", "")[:300], "score": r.get("score", 0.0)}
                for r in results[:k]
            ],
            "count": len(results),
        }
    except Exception as exc:
        logger.warning("[agent.search] 搜索失败: %s", exc)
        return {"query": sub_query, "results": [], "count": 0, "error": str(exc)}


async def _aggregate_results_tool(
    original_query: str,
    search_results: list[dict],
    model_name: str | None = None,
) -> dict:
    """Tool: 汇总多轮检索结果."""
    try:
        results_text = "\n\n---\n\n".join(
            f"子问题: {r.get('query', '')}\n"
            f"结果: {[d.get('content', '')[:200] for d in r.get('results', [])]}"
            for r in search_results
        )

        llm = _get_agent_llm(model_name)
        llm_start = time.monotonic()
        resp = await llm.client.chat.completions.create(
            model=llm.model_id,
            messages=[
                {"role": "system", "content": AGGREGATE_SYSTEM},
                {"role": "user", "content": (
                    f"原始查询: {original_query}\n\n"
                    f"各子问题检索结果:\n{results_text}"
                )},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        track_llm_call(
            name="agent_aggregate_llm",
            model=llm.model_id,
            start_time=llm_start,
            response=resp,
        )

        return {
            "summary": resp.choices[0].message.content.strip(),
            "source_count": len(search_results),
        }
    except Exception as exc:
        logger.warning("[agent.aggregate] 汇总失败: %s", exc)
        return {"summary": "", "source_count": len(search_results), "error": str(exc)}


@trace_rag_node(name="analytical_agent")
async def run_analytical_agent(state: dict) -> dict:
    """Analytical Agent 节点入口（含超时和 fallback）。

    直接手写 ReAct 循环而非嵌入子图，以便控制超时和 fallback。
    """

    query = state.get("query", "")
    documents = state.get("documents") or []

    try:
        result = await asyncio.wait_for(
            _agent_loop(query, documents, model_name=state.get("model_name")),
            timeout=AGENT_TIMEOUT_S,
        )
        langfuse_context.update_current_observation(
            node="analytical_agent",
            sub_query_count=len(result.get("agent_sub_queries", [query])),
            iterations=result.get("agent_iterations", 1),
            fallback=result.get("agent_fallback", False),
        )
        return result
    except asyncio.TimeoutError:
        logger.warning("[analytical_agent] 超时 (%.1fs) → fallback complex pipeline", AGENT_TIMEOUT_S)
        langfuse_context.update_current_observation(
            node="analytical_agent",
            fallback=True,
            error=f"timeout after {AGENT_TIMEOUT_S}s",
        )
        return {
            "agent_fallback": True,
            "agent_iterations": AGENT_MAX_ITER,
            "errors": [f"analytical_agent: timeout after {AGENT_TIMEOUT_S}s"],
        }
    except Exception as exc:
        logger.warning("[analytical_agent] 异常: %s → fallback", exc)
        langfuse_context.update_current_observation(
            node="analytical_agent",
            fallback=True,
            error=str(exc),
        )
        return {
            "agent_fallback": True,
            "agent_iterations": 0,
            "errors": [f"analytical_agent: {exc}"],
        }


async def _agent_loop(
    query: str,
    existing_docs: list[dict],
    model_name: str | None = None,
) -> dict:
    """Analytical Agent 主循环：decompose → search → aggregate."""

    # Step 1: Decompose
    decomp = await _decompose_query_tool(query, model_name=model_name)
    sub_queries = decomp.get("sub_queries", [query])

    # Step 2: Parallel search for each sub-query
    search_tasks = [_search_docs_tool(sq) for sq in sub_queries]
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    clean_results = []
    all_docs = list(existing_docs)

    for i, res in enumerate(search_results):
        if isinstance(res, Exception):
            clean_results.append({"query": sub_queries[i], "results": [], "count": 0})
        else:
            clean_results.append(res)
            for r in res.get("results", []):
                all_docs.append({"content": r.get("content", ""), "score": r.get("score", 0.0)})

    # Step 3: Aggregate
    agg = await _aggregate_results_tool(query, clean_results, model_name=model_name)
    summary = agg.get("summary", "")

    if summary:
        all_docs.insert(0, {
            "content": summary,
            "source": "agent_aggregated",
            "score": 1.0,
            "rerank_score": 1.0,
        })

    return {
        "documents": all_docs,
        "agent_iterations": 1,
        "agent_sub_queries": sub_queries,
        "agent_fallback": False,
    }
