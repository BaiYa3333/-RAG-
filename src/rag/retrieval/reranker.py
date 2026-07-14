"""重排序 — DashScope qwen3-rerank Cross-encoder (fallback → LLM listwise → identity sort).

gte-rerank 已于 2026-05-30 下线，迁移至 qwen3-rerank (专用文本 reranker)。
API 格式: query 需包装为 {"text": "..."}, documents 需包装为 [{"text": "..."}, ...]。

关键词增强: 对 Cross-encoder 分数做关键词匹配加权，提升产品名/实体名等
精确词汇的区分度，弥补纯语义模型的短板。权重: 0.85 × API + 0.15 × keyword。
"""

import asyncio
import json
import logging
import time

import httpx
from src.observability.langfuse_context import langfuse_context

from src.config import settings
from src.rag.retrieval.keyword_boost import keyword_match_score
from src.observability.decorators import trace_rag_node
from src.observability.llm_tracker import track_llm_call

logger = logging.getLogger(__name__)

RERANK_API_URL = settings.rerank_api_url
RERANK_MODEL = settings.rerank_model
API_KEY = settings.qwen_api_key
RERANK_TIMEOUT = settings.rerank_timeout_s

# LLM 重排序 batch size（每次 LLM 调用评测的文档数）
LLM_RERANK_BATCH = 10

# 关键词增强权重 (API_score * API_WEIGHT + keyword_score * KW_WEIGHT)
# 保持保守的混合比例：语义模型为主，关键词辅助精确匹配
API_WEIGHT = 0.85
KW_WEIGHT = 0.15


def _blend_scores(api_scores: dict[int, float], documents: list[dict],
                  query: str) -> dict[int, float]:
    """融合 API 分数与关键词匹配分数。

    Args:
        api_scores: {doc_index: relevance_score} 来自 API
        documents: 原始文档列表
        query: 查询文本

    Returns:
        {doc_index: blended_score}
    """
    blended = {}
    for idx, api_score in api_scores.items():
        if idx < len(documents):
            content = documents[idx].get("content", "")
            kw_score = keyword_match_score(query=query, doc_content=content)
            # 关键词权重混合
            blended[idx] = API_WEIGHT * api_score + KW_WEIGHT * kw_score
        else:
            blended[idx] = api_score
    return blended


@trace_rag_node(name="rerank")
async def rerank(
    query: str,
    documents: list[dict],
    top_k: int | None = None,
    model_name: str | None = None,
) -> list[dict]:
    """Cross-encoder 重排序 via DashScope Rerank API (qwen3-rerank).

    Args:
        query: 查询文本
        documents: 候选文档列表，每个包含 "content" 字段
        top_k: 返回数量，默认 settings.rerank_top_k (5)

    Returns:
        按 relevance_score 降序排列的 top_k 文档列表。
        API 失败时 fallback 到 identity 排序。
    """

    if not documents:
        return []

    k = top_k or settings.rerank_top_k

    # 提取文档文本，包装为 DashScope Rerank API 格式
    doc_dicts = [{"text": d.get("content", "")} for d in documents]

    try:
        async with httpx.AsyncClient(timeout=RERANK_TIMEOUT) as client:
            resp = await client.post(
                RERANK_API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": RERANK_MODEL,
                    "input": {
                        "query": {"text": query},
                        "documents": doc_dicts,
                    },
                    "parameters": {
                        "top_n": k,
                        "return_documents": False,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # 解析 DashScope 响应: output.results = [{index, relevance_score}, ...]
        results = data.get("output", {}).get("results", [])
        if not results:
            logger.warning("[rerank] API 返回空结果，降级到 identity + keyword 排序")
            return _identity_rerank(documents, k, query)

        # 提取 API 分数并做关键词增强
        api_scores = {}
        for item in results:
            idx = item.get("index", 0)
            score = item.get("relevance_score", 0.0)
            if idx < len(documents):
                api_scores[idx] = score

        # 关键词融合
        blended = _blend_scores(api_scores, documents, query)

        # 按混合分数降序取 top_k
        sorted_indices = sorted(blended, key=lambda i: blended[i], reverse=True)

        reranked = []
        for idx in sorted_indices[:k]:
            doc = {**documents[idx], "rerank_score": round(blended[idx], 6), "_rerank_method": "api"}
            reranked.append(doc)

        logger.info("[rerank] qwen3-rerank + keyword boost 完成: %d docs → top %d",
                    len(documents), len(reranked))
        langfuse_context.update_current_observation(
            rerank_method="api",
            input_count=len(documents),
            output_count=len(reranked),
            model=RERANK_MODEL,
        )
        return reranked

    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.warning("[rerank] API 请求失败: %s → 降级 LLM 重排序", exc)
        try:
            result = await _llm_rerank(query, documents, k, model_name=model_name)
            langfuse_context.update_current_observation(
                rerank_method="llm",
                input_count=len(documents),
                output_count=len(result),
                reason="api_failed",
            )
            return result
        except Exception as llm_exc:
            logger.warning("[rerank] LLM 重排序也失败: %s → identity + keyword 排序", llm_exc)
            result = _identity_rerank(documents, k, query)
            langfuse_context.update_current_observation(
                rerank_method="identity",
                input_count=len(documents),
                output_count=len(result),
                reason="api_and_llm_failed",
            )
            return result
    except Exception as exc:
        logger.warning("[rerank] 未知错误: %s → 降级 LLM 重排序", exc)
        try:
            result = await _llm_rerank(query, documents, k, model_name=model_name)
            langfuse_context.update_current_observation(
                rerank_method="llm",
                input_count=len(documents),
                output_count=len(result),
                reason="unknown_error",
            )
            return result
        except Exception as llm_exc:
            logger.warning("[rerank] LLM 重排序也失败: %s → identity + keyword 排序", llm_exc)
            result = _identity_rerank(documents, k, query)
            langfuse_context.update_current_observation(
                rerank_method="identity",
                input_count=len(documents),
                output_count=len(result),
                reason="unknown_error_llm_failed",
            )
            return result


def _identity_rerank(documents: list[dict], top_k: int, query: str = "") -> list[dict]:
    """按已有分数降级排序 (identity rerank fallback).

    如果提供了 query，用关键词匹配分数与已有分数混合排序。
    """
    if query:
        # 混合已有分数 + 关键词匹配
        blended = {}
        for i, d in enumerate(documents):
            base = d.get("rrf_score", d.get("score", 0))
            kw_score = keyword_match_score(query=query, doc_content=d.get("content", ""))
            blended[i] = 0.7 * base + 0.3 * kw_score
        sorted_indices = sorted(blended, key=lambda i: blended[i], reverse=True)
        return [
            {**documents[i], "rerank_score": round(blended[i], 6), "_rerank_method": "identity"}
            for i in sorted_indices[:top_k]
        ]

    return [
        {**doc, "_rerank_method": "identity"}
        for doc in sorted(
            documents,
            key=lambda d: d.get("rrf_score", d.get("score", 0)),
            reverse=True,
        )[:top_k]
    ]


async def _llm_rerank(
    query: str,
    documents: list[dict],
    top_k: int,
    model_name: str | None = None,
) -> list[dict]:
    """LLM listwise 重排序 — API 专用 rerank 不可用时的 fallback.

    将文档分批交给 LLM 打分（0-10），按分数降序返回 top_k。
    """
    from src.llm.factory import create_llm

    if len(documents) <= top_k:
        return [{**doc, "_rerank_method": "llm"} for doc in documents[:top_k]]

    # 分批评测
    batches = [
        documents[i : i + LLM_RERANK_BATCH]
        for i in range(0, len(documents), LLM_RERANK_BATCH)
    ]

    scored_docs: list[dict] = []

    for batch in batches:
        doc_texts = "\n\n---\n\n".join(
            f"[文档{i+1}]\n{d.get('content', '')[:300]}"
            for i, d in enumerate(batch)
        )
        prompt = f"""评估以下文档与查询的相关性，为每个文档打分(0-10，10=完全相关)。

查询: {query}

{doc_texts}

为每个文档输出 JSON 数组：[{{"doc_idx": 1, "score": 8.5}}, ...]
只输出 JSON，不输出其他内容。"""

        try:
            llm = create_llm(model_name=model_name)
            llm_start = time.monotonic()
            resp = await llm.client.chat.completions.create(
                model=llm.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            track_llm_call(
                name="rerank_llm_batch",
                model=llm.model_id,
                start_time=llm_start,
                response=resp,
            )
            raw = resp.choices[0].message.content.strip()

            # 解析 JSON
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            scores_data = json.loads(raw)
            score_map = {item["doc_idx"]: item["score"] for item in scores_data}

            for i, doc in enumerate(batch):
                doc_copy = {**doc, "rerank_score": score_map.get(i + 1, 0.0), "_rerank_method": "llm"}
                scored_docs.append(doc_copy)

        except Exception as exc:
            logger.warning(
                "[llm_rerank] batch scoring 失败: %s → 使用 identity score",
                exc,
            )
            for doc in batch:
                doc_copy = {
                    **doc,
                    "rerank_score": doc.get("rrf_score", doc.get("score", 0)),
                    "_rerank_method": "identity",
                }
                scored_docs.append(doc_copy)

    # 按 rerank_score 降序
    scored_docs.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
    logger.info("[llm_rerank] LLM 重排序完成: %d docs → top %d", len(scored_docs), top_k)
    return scored_docs[:top_k]
