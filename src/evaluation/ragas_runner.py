"""RAGAS 评估流程编排器 — 串联测试集构造→RAG 查询→指标计算→持久化."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.evaluation.metrics import compute_ragas_metrics, EvalMetrics
from src.evaluation.testset_generator import generate_testset, TestsetItem
from src.evaluation.history import save_evaluation_result
from src.utils.logger import logger


class EvaluationRunner:
    """RAGAS 评估流程编排器。

    在 FastAPI lifespan 中初始化一次，持有共享依赖引用，
    供 /eval/run 端点复用。

    支持两种模式：
    - auto: 自动从知识库生成测试集
    - manual: 使用用户提供的测试集
    """

    def __init__(self, llm_client, vector_store, doc_store):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.doc_store = doc_store

    async def evaluate_single(self, item: dict) -> dict:
        """Evaluate a single test case and return metric scores.

        Args:
            item: dict with keys:
                - question (str): the query/question
                - ground_truth (str): expected answer
                - reference_contexts (list[str]): relevant context passages
                - answer (str, optional): pre-generated answer. If absent,
                  an answer is generated via the configured LLM.

        Returns:
            dict with RAGAS metric scores (faithfulness, answer_relevancy,
            context_precision, context_recall, avg_score).
        """
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        reference_contexts = item.get("reference_contexts", [])
        answer = item.get("answer")

        # Generate answer if not provided
        if not answer:
            try:
                from src.llm.factory import LLMClient

                context_snippet = "\n".join(
                    c[:500] for c in (reference_contexts or [])[:3]
                )
                if isinstance(self.llm_client, LLMClient):
                    resp = await self.llm_client.client.chat.completions.create(
                        model=self.llm_client.model_id,
                        messages=[
                            {
                                "role": "system",
                                "content": "基于提供的上下文回答问题。",
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"上下文:\n{context_snippet}\n\n"
                                    f"问题: {question}"
                                ),
                            },
                        ],
                        temperature=0.3,
                        max_tokens=512,
                    )
                    answer = resp.choices[0].message.content or ""
                else:
                    answer = "N/A (LLM client unavailable)"
            except Exception as e:
                logger.warning("evaluate_single_generate_failed", error=str(e))
                answer = "N/A (LLM generation failed)"

        # Compute RAGAS metrics
        metrics = await compute_ragas_metrics(
            query=question,
            contexts=reference_contexts,
            answer=answer or "",
            ground_truth=ground_truth,
        )

        return {
            "question": question,
            "ground_truth": ground_truth,
            "answer": (answer or "")[:300],
            "contexts_count": len(reference_contexts),
            **metrics.to_dict(),
        }

    async def run(
        self,
        testset_source: str = "auto",
        document_ids: list[str] | None = None,
        testset: list[dict] | None = None,
        testset_size: int = 10,
        graph=None,
        kb_ids: list[str] | None = None,
    ) -> dict:
        """执行完整 RAGAS 评估流程。

        Args:
            testset_source: "auto" 或 "manual"
            document_ids: auto 模式下指定的文档 ID
            testset: manual 模式下用户提供的测试集 [{"question", "ground_truth"}]
            testset_size: auto 模式下生成的测试集大小
            graph: LangGraph 预编译 workflow（从 app.state.graph 获取）
            kb_ids: 目标知识库 ID 列表（为空时使用默认集合）

        Returns:
            评估结果字典，含 evaluation_id 和四项指标分数
        """
        evaluation_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        # 1. 构造测试集
        if testset_source == "auto":
            items = await generate_testset(
                doc_store=self.doc_store,
                vector_store=self.vector_store,
                llm_client=self.llm_client,
                document_ids=document_ids,
                size=testset_size,
            )
        elif testset_source == "manual":
            if not testset:
                raise ValueError("manual mode requires 'testset' parameter")
            items = [
                TestsetItem(
                    question=t.get("question", ""),
                    ground_truth=t.get("ground_truth", ""),
                    reference_contexts=t.get("reference_contexts", []),
                )
                for t in testset
            ]
        else:
            raise ValueError(f"Invalid testset_source: {testset_source}")

        if not items:
            logger.warning("eval_no_testset_items", source=testset_source)
            empty_metrics = EvalMetrics(0.0, 0.0, 0.0, 0.0)
            return {
                "evaluation_id": evaluation_id,
                "testset_size": 0,
                "created_at": started_at.isoformat(),
                **empty_metrics.to_dict(),
            }

        logger.info("eval_started", evaluation_id=evaluation_id, testset_size=len(items))

        # 2. 逐条执行 RAG 查询 → 指标计算
        all_metrics: list[EvalMetrics] = []
        details: list[dict] = []

        for i, item in enumerate(items):
            try:
                # 执行 RAG 查询
                if graph:
                    from src.graph.state import RAGState, initial_state
                    initial = initial_state(item.question, chat_history=[], kb_ids=kb_ids or [])
                    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                    final_state = await graph.ainvoke(initial, config)
                    answer = final_state.get("answer", "")
                    used_contexts = final_state.get("used_contexts") or []
                    if used_contexts:
                        contexts = [c for c in used_contexts if c]
                    else:
                        contexts = [
                            d.get("content", "")
                            for d in final_state.get("documents", [])
                            if d.get("content", "")
                        ]
                else:
                    # fallback: 直接用 LLM 生成答案（无 graph 场景）
                    contexts = item.reference_contexts or []
                    try:
                        from src.llm.factory import LLMClient
                        if isinstance(self.llm_client, LLMClient):
                            resp = await self.llm_client.client.chat.completions.create(
                                model=self.llm_client.model_id,
                                messages=[
                                    {"role": "system", "content": "基于提供的上下文回答问题。"},
                                    {"role": "user", "content": f"上下文:\n{chr(10).join(contexts[:3])}\n\n问题: {item.question}"},
                                ],
                                temperature=0.3,
                                max_tokens=512,
                            )
                            answer = resp.choices[0].message.content or ""
                        else:
                            answer = "N/A (LLM client unavailable)"
                    except Exception:
                        answer = "N/A (LLM generation failed)"

                # 计算 RAGAS 指标
                metrics = await compute_ragas_metrics(
                    query=item.question,
                    contexts=contexts,
                    answer=answer,
                    ground_truth=item.ground_truth,
                )
                all_metrics.append(metrics)

                details.append({
                    "question": item.question,
                    "ground_truth": item.ground_truth,
                    "answer": answer[:300],
                    "contexts_count": len(contexts),
                    **metrics.to_dict(),
                })

                logger.info(
                    "eval_item_completed",
                    evaluation_id=evaluation_id,
                    idx=i + 1,
                    total=len(items),
                    avg=metrics.avg_score,
                )

            except Exception as e:
                logger.error(
                    "eval_item_failed",
                    evaluation_id=evaluation_id,
                    idx=i,
                    error=str(e),
                )
                # 单条失败不阻塞整体
                details.append({
                    "question": item.question,
                    "ground_truth": item.ground_truth,
                    "error": str(e),
                })

        # 3. 聚合指标
        if all_metrics:
            agg = EvalMetrics(
                faithfulness=round(sum(m.faithfulness for m in all_metrics) / len(all_metrics), 4),
                answer_relevancy=round(sum(m.answer_relevancy for m in all_metrics) / len(all_metrics), 4),
                context_precision=round(sum(m.context_precision for m in all_metrics) / len(all_metrics), 4),
                context_recall=round(sum(m.context_recall for m in all_metrics) / len(all_metrics), 4),
            )
        else:
            agg = EvalMetrics(0.0, 0.0, 0.0, 0.0)

        # 4. 持久化
        try:
            await save_evaluation_result(
                doc_store=self.doc_store,
                evaluation_id=evaluation_id,
                faithfulness=agg.faithfulness,
                answer_relevancy=agg.answer_relevancy,
                context_precision=agg.context_precision,
                context_recall=agg.context_recall,
                avg_score=agg.avg_score,
                testset_size=len(items),
                details=details,
                created_at=started_at,
            )
        except Exception as e:
            logger.error("eval_persist_failed", evaluation_id=evaluation_id, error=str(e))

        result = {
            "evaluation_id": evaluation_id,
            "testset_size": len(items),
            "created_at": started_at.isoformat(),
            **agg.to_dict(),
        }

        logger.info("eval_completed", **result)
        return result
