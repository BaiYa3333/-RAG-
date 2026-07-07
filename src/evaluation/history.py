"""评估历史持久化 — PostgreSQL evaluation_results 表读写."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from src.utils.logger import logger


async def save_evaluation_result(
    doc_store,
    evaluation_id: str,
    faithfulness: float,
    answer_relevancy: float,
    context_precision: float,
    context_recall: float,
    avg_score: float,
    testset_size: int,
    details: list[dict],
    created_at: datetime | None = None,
) -> str:
    """保存评估结果到 evaluation_results 表。

    使用实际表结构 (docker/init.sql):
    - id: auto UUID
    - run_id: 评估 run uuid (evaluation_id)
    - testset_id: 测试集 uuid
    - config_label: 评估配置标签
    - faithfulness/answer_relevancy/context_precision/context_recall: 4 项指标
    - config_snapshot: JSONB (avg_score, testset_size, details)
    - created_at

    Returns:
        evaluation_id
    """
    created = created_at or datetime.now(timezone.utc)
    testset_uuid = str(uuid.uuid4())

    config_snapshot = json.dumps({
        "avg_score": avg_score,
        "testset_size": testset_size,
        "details": details,
    }, ensure_ascii=False)

    query = """
        INSERT INTO evaluation_results
            (run_id, testset_id, config_label, faithfulness, answer_relevancy,
             context_precision, context_recall, config_snapshot, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id
    """
    try:
        result = await doc_store.fetchval(
            query,
            evaluation_id,    # run_id
            testset_uuid,     # testset_id
            "ragas_eval",     # config_label
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            config_snapshot,
            created,
        )
        logger.info("eval_result_saved", evaluation_id=evaluation_id, db_id=str(result))
        return evaluation_id
    except Exception as e:
        logger.error("eval_result_save_failed", evaluation_id=evaluation_id, error=str(e))
        raise


async def get_evaluation_results(
    doc_store,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """查询历史评估结果，按时间降序。

    Returns:
        {"total": int, "results": [dict, ...]}
    """
    try:
        total = await doc_store.fetchval("SELECT COUNT(*) FROM evaluation_results") or 0

        query = """
            SELECT id, run_id, config_label, faithfulness, answer_relevancy,
                   context_precision, context_recall, config_snapshot, created_at
            FROM evaluation_results
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        """
        rows = await doc_store.fetch(query, limit, offset)

        results = []
        for row in rows:
            created_at = row.get("created_at")
            if isinstance(created_at, str):
                created_str = created_at
            elif created_at is not None:
                created_str = created_at.isoformat()
            else:
                created_str = ""

            # 从 config_snapshot 提取附加字段
            snapshot = row.get("config_snapshot") or {}
            if isinstance(snapshot, str):
                try:
                    snapshot = json.loads(snapshot)
                except json.JSONDecodeError:
                    snapshot = {}

            results.append({
                "evaluation_id": str(row["run_id"]),
                "faithfulness": float(row["faithfulness"] or 0),
                "answer_relevancy": float(row["answer_relevancy"] or 0),
                "context_precision": float(row["context_precision"] or 0),
                "context_recall": float(row["context_recall"] or 0),
                "avg_score": snapshot.get("avg_score", 0.0),
                "testset_size": snapshot.get("testset_size", 0),
                "details": snapshot.get("details", []),
                "created_at": created_str,
            })

        return {"total": total, "results": results}

    except Exception as e:
        logger.error("eval_results_query_failed", error=str(e))
        raise


async def get_latest_evaluation(doc_store) -> dict | None:
    """获取最近一次评估结果."""
    query = """
        SELECT id, run_id, config_label, faithfulness, answer_relevancy,
               context_precision, context_recall, config_snapshot, created_at
        FROM evaluation_results
        ORDER BY created_at DESC
        LIMIT 1
    """
    row = await doc_store.fetchrow(query)
    if not row:
        return None

    created_at = row.get("created_at")
    if isinstance(created_at, str):
        created_str = created_at
    elif created_at is not None:
        created_str = created_at.isoformat()
    else:
        created_str = ""

    snapshot = row.get("config_snapshot") or {}
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except json.JSONDecodeError:
            snapshot = {}

    return {
        "evaluation_id": str(row["run_id"]),
        "faithfulness": float(row["faithfulness"] or 0),
        "answer_relevancy": float(row["answer_relevancy"] or 0),
        "context_precision": float(row["context_precision"] or 0),
        "context_recall": float(row["context_recall"] or 0),
        "avg_score": snapshot.get("avg_score", 0.0),
        "testset_size": snapshot.get("testset_size", 0),
        "details": snapshot.get("details", []),
        "created_at": created_str,
    }
