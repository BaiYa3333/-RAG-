"""评估 API 路由 — POST /eval/run + GET /eval/results + GET /eval/dashboard + POST /eval/stream."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from src.api.schemas import (
    EvalRequest,
    EvalResponse,
    EvalResultsResponse,
)
from src.evaluation.ragas_runner import EvaluationRunner
from src.evaluation.history import get_evaluation_results, get_latest_evaluation
from src.evaluation.visualize import (
    generate_radar_chart,
    generate_detail_table,
    build_dashboard_html,
)
from src.utils.logger import logger

evaluation_router = APIRouter(prefix="/eval", tags=["evaluation"])


@evaluation_router.post("/run", response_model=EvalResponse, summary="触发 RAGAS 评估")
async def run_evaluation(req: EvalRequest, request: Request):
    """提交评估请求，支持 auto（自动生成测试集）和 manual（手动提供测试集）两种模式。"""
    runner: EvaluationRunner | None = getattr(
        request.app.state, "evaluation_runner", None
    )
    if runner is None:
        raise HTTPException(
            status_code=503,
            detail="评估服务不可用：EvaluationRunner 未初始化",
        )

    graph = getattr(request.app.state, "graph", None)

    try:
        result = await runner.run(
            testset_source=req.testset_source,
            document_ids=req.document_ids,
            testset=[t.model_dump() for t in req.testset] if req.testset else None,
            testset_size=req.testset_size,
            graph=graph,
            kb_ids=req.kb_ids,
        )
        return EvalResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("eval_run_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"评估执行失败: {e}")


@evaluation_router.get("/results", response_model=EvalResultsResponse, summary="查询评估历史")
async def query_evaluation_results(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100, description="返回数量"),
    offset: int = Query(default=0, ge=0, description="偏移量"),
):
    """查询历史评估结果，按创建时间降序排列。"""
    runner: EvaluationRunner | None = getattr(
        request.app.state, "evaluation_runner", None
    )
    if runner is None:
        raise HTTPException(
            status_code=503,
            detail="评估服务不可用：EvaluationRunner 未初始化",
        )

    try:
        data = await get_evaluation_results(
            doc_store=runner.doc_store,
            limit=limit,
            offset=offset,
        )
        results = [EvalResponse(**r) for r in data["results"]]
        return EvalResultsResponse(total=data["total"], results=results)

    except Exception as e:
        logger.error("eval_results_query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"查询评估结果失败: {e}")


@evaluation_router.get("/dashboard", summary="RAGAS 评估可视化 Dashboard")
async def eval_dashboard(request: Request):
    """返回自包含 HTML 页面，展示雷达图和明细表。"""
    runner: EvaluationRunner | None = getattr(
        request.app.state, "evaluation_runner", None
    )
    if runner is None:
        raise HTTPException(
            status_code=503,
            detail="评估服务不可用：EvaluationRunner 未初始化",
        )

    try:
        # 读取评估历史
        data = await get_evaluation_results(
            doc_store=runner.doc_store,
            limit=20,
            offset=0,
        )
        history = data["results"]
        latest = history[0] if history else None

        # 生成两张图表：雷达图 + 明细表
        if latest:
            radar = generate_radar_chart(latest)
        else:
            radar = generate_radar_chart({})

        table = generate_detail_table(history)

        last_updated = latest.get("created_at", "") if latest else ""

        html = build_dashboard_html(
            radar_fig=radar,
            table_fig=table,
            last_updated=last_updated,
        )

        return HTMLResponse(content=html, status_code=200)

    except Exception as e:
        logger.error("eval_dashboard_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"生成 Dashboard 失败: {e}")


@evaluation_router.post("/evaluate/stream", summary="Stream evaluation progress (SSE)")
async def evaluate_stream(body: EvalRequest, request: Request):
    """Run RAGAS evaluation and stream per-question progress via SSE.

    Events: eval_question_start | eval_question_complete | eval_done | error
    """
    from src.api.sse_events import (
        eval_question_start_event,
        eval_question_complete_event,
        eval_done_event,
        error_event,
    )
    from src.utils.logger import logger

    eval_runner = getattr(request.app.state, "evaluation_runner", None)
    if eval_runner is None:
        async def err():
            yield f"event: error\ndata: {json.dumps(error_event(message='Evaluation runner not available'), ensure_ascii=False)}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    async def event_generator():
        try:
            if body.testset_source == "manual" and body.testset:
                testset = [t.model_dump() if hasattr(t, 'model_dump') else t for t in body.testset]
            else:
                testset = await eval_runner.generate_testset(
                    document_ids=body.document_ids,
                    testset_size=body.testset_size,
                )

            total = len(testset)
            if total == 0:
                yield f"event: error\ndata: {json.dumps(error_event(message='Empty testset'), ensure_ascii=False)}\n\n"
                return

            all_scores = {
                "faithfulness": [],
                "answer_relevancy": [],
                "context_precision": [],
                "context_recall": [],
            }

            for i, item in enumerate(testset):
                q = item.get("question", "")[:100] if isinstance(item, dict) else getattr(item, "question", "")[:100]
                yield f"event: eval_question_start\ndata: {json.dumps(eval_question_start_event(i, total, q), ensure_ascii=False)}\n\n"

                result = await eval_runner.evaluate_single(item)
                scores = result.get("scores", result)

                all_scores["faithfulness"].append(scores.get("faithfulness", 0))
                all_scores["answer_relevancy"].append(scores.get("answer_relevancy", 0))
                all_scores["context_precision"].append(scores.get("context_precision", 0))
                all_scores["context_recall"].append(scores.get("context_recall", 0))

                question_complete = eval_question_complete_event(
                    i,
                    faithfulness=scores.get("faithfulness", 0),
                    answer_relevancy=scores.get("answer_relevancy", 0),
                    context_precision=scores.get("context_precision", 0),
                    context_recall=scores.get("context_recall", 0),
                )
                yield f"event: eval_question_complete\ndata: {json.dumps(question_complete, ensure_ascii=False)}\n\n"

            # Final aggregate
            avg = {k: sum(v) / len(v) if v else 0 for k, v in all_scores.items()}
            avg_score_val = sum(avg.values()) / 4
            done_data = eval_done_event(
                faithfulness=avg["faithfulness"],
                answer_relevancy=avg["answer_relevancy"],
                context_precision=avg["context_precision"],
                context_recall=avg["context_recall"],
                avg_score=avg_score_val,
            )
            yield f"event: eval_done\ndata: {json.dumps(done_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error("evaluate_stream_failed", error=str(e))
            yield f"event: error\ndata: {json.dumps(error_event(message=str(e)), ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
