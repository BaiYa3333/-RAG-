"""RAGAS 指标计算 — faithfulness / answer_relevancy / context_precision / context_recall."""

import sys
import types
from dataclasses import dataclass

from openai import OpenAI

from src.config import settings
from src.utils.logger import logger

# ── RAGAS 兼容性补丁 ──────────────────────────────────────────
# ragas 0.4.3 内部仍引用 langchain_community.chat_models.vertexai，
# 但 langchain-community >= 0.4.0 已移除该路径。
# 此处创建一个 shim 模块防止 ImportError（RAGAS 对此仅作 hasattr 检查）。
_PATCH_KEY = "langchain_community.chat_models.vertexai"
if _PATCH_KEY not in sys.modules:
    try:
        from langchain_community.chat_models import vertexai  # noqa: F401
    except ImportError:
        _shim = types.ModuleType(_PATCH_KEY)
        _shim.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[_PATCH_KEY] = _shim
        import langchain_community.chat_models
        langchain_community.chat_models.vertexai = _shim
        logger.info("ragas_vertexai_shim_applied")


# ── DashScope Embedding 适配器 ─────────────────────────────────
# LangChain 的 OpenAIEmbeddings 发送的请求格式与 DashScope 兼容端点
# 不兼容（DashScope 报 "contents is neither str nor list of str"）。
# 此适配器使用原生 openai.OpenAI client，提供 embed_query / embed_documents
# 两个方法，满足 ragas 旧版 API 的 Embedding 接口要求。

class DashScopeEmbeddings:
    """DashScope 兼容 OpenAI 的 Embedding 适配器。

    使用原生 OpenAI client 调用 DashScope 兼容端点，
    避免 LangChain OpenAIEmbeddings 的请求格式问题。
    """

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v4"):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        """嵌入单条查询文本."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文档文本."""
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


@dataclass
class EvalMetrics:
    """四项 RAGAS 评估指标结果 (0.0–1.0)."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return round(
            (self.faithfulness
             + self.answer_relevancy
             + self.context_precision
             + self.context_recall)
            / 4,
            4,
        )

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "avg_score": self.avg_score,
        }


async def compute_ragas_metrics(
    query: str,
    contexts: list[str],
    answer: str,
    ground_truth: str | None = None,
) -> EvalMetrics:
    """调用 RAGAS 库计算四项核心指标。

    Args:
        query: 用户查询
        contexts: 检索到的上下文列表
        answer: LLM 生成的答案
        ground_truth: 参考答案（可选，用于 context_recall）

    Returns:
        EvalMetrics 含四项指标 (0.0-1.0)
    """
    import asyncio

    def _run_ragas_sync() -> dict:
        """在独立线程中运行 RAGAS evaluate（避开 uvloop 嵌套事件循环限制）."""

        # ── LLM: DeepSeek (OpenAI 兼容) via ragas llm_factory ──
        from ragas.llms import llm_factory

        llm_client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        eval_llm = llm_factory("deepseek-chat", client=llm_client, max_tokens=4096)

        # ── Embeddings: 自定义 DashScope 适配器 ──
        eval_emb = DashScopeEmbeddings(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
        )

        # ── 指标: 旧版 ragas.metrics API (兼容 Metric 基类) ──
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        metrics_to_use = [faithfulness, answer_relevancy, context_precision]
        if ground_truth:
            metrics_to_use.append(context_recall)

        eval_data: dict = {
            "question": [query],
            "contexts": [contexts],
            "answer": [answer],
        }
        if ground_truth:
            eval_data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(eval_data)

        result = evaluate(
            dataset,
            metrics=metrics_to_use,
            llm=eval_llm,
            embeddings=eval_emb,
        )

        # EvaluationResult → DataFrame → dict
        df = result.to_pandas()
        row = df.iloc[0].to_dict() if len(df) > 0 else {}

        def _safe_float(key: str) -> float:
            val = row.get(key, 0.0)
            try:
                f = float(val)
                return f if f == f else 0.0  # NaN → 0.0
            except (ValueError, TypeError):
                return 0.0

        return {
            "faithfulness": _safe_float("faithfulness"),
            "answer_relevancy": _safe_float("answer_relevancy"),
            "context_precision": _safe_float("context_precision"),
            "context_recall": _safe_float("context_recall") if ground_truth else 0.0,
        }

    try:
        # RAGAS evaluate() 内部使用 asyncio.run()，与 uvloop 冲突。
        # 在线程池中运行以使用独立的默认事件循环。
        result = await asyncio.to_thread(_run_ragas_sync)

        faith = round(result["faithfulness"], 4)
        relevancy = round(result["answer_relevancy"], 4)
        precision = round(result["context_precision"], 4)
        recall = round(result["context_recall"], 4) if ground_truth else 0.0

        logger.info(
            "ragas_metrics_computed",
            query=query[:60],
            faithfulness=faith,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
        )

        return EvalMetrics(
            faithfulness=faith,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
        )

    except ImportError as e:
        logger.error("ragas_import_error", error=str(e))
        raise RuntimeError(
            "RAGAS library not available. Install with: pip install ragas datasets"
        ) from e
    except Exception as e:
        logger.error("ragas_compute_error", error=str(e))
        return EvalMetrics(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=0.0,
        )
