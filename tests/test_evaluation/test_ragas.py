"""测试 RAGAS 指标计算模块."""

import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.evaluation.metrics import EvalMetrics


class TestEvalMetrics:
    """EvalMetrics dataclass 单元测试."""

    def test_to_dict_returns_correct_format(self):
        m = EvalMetrics(
            faithfulness=0.85,
            answer_relevancy=0.78,
            context_precision=0.82,
            context_recall=0.91,
        )
        d = m.to_dict()
        assert d["faithfulness"] == 0.85
        assert d["answer_relevancy"] == 0.78
        assert d["context_precision"] == 0.82
        assert d["context_recall"] == 0.91
        expected_avg = round((0.85 + 0.78 + 0.82 + 0.91) / 4, 4)
        assert d["avg_score"] == expected_avg

    def test_avg_score_correct(self):
        m = EvalMetrics(0.5, 0.5, 0.5, 0.5)
        assert m.avg_score == 0.5

    def test_all_zeros(self):
        m = EvalMetrics(0.0, 0.0, 0.0, 0.0)
        assert m.avg_score == 0.0


class TestComputeRagasMetrics:
    """RAGAS 指标计算测试 — mock ragas 在 sys.modules 中."""

    @pytest.fixture(autouse=True)
    def setup_mock_ragas(self):
        """在每个测试前 mock ragas + datasets 模块，支持子包导入。"""
        # 创建 ragas 包及其子模块
        import types

        mock_ragas = types.ModuleType("ragas")
        mock_ragas.__path__ = ["ragas"]  # 标记为 package
        mock_ragas.evaluate = MagicMock()

        mock_ragas_metrics = types.ModuleType("ragas.metrics")
        mock_ragas_metrics.faithfulness = MagicMock()
        mock_ragas_metrics.answer_relevancy = MagicMock()
        mock_ragas_metrics.context_precision = MagicMock()
        mock_ragas_metrics.context_recall = MagicMock()

        # ragas.llms — metrics.py _run_ragas_sync 中 from ragas.llms import llm_factory
        mock_ragas_llms = types.ModuleType("ragas.llms")
        mock_ragas_llms.llm_factory = MagicMock()

        mock_datasets = types.ModuleType("datasets")
        mock_datasets.Dataset = MagicMock()
        mock_datasets.Dataset.from_dict = MagicMock()

        # 保存原始模块
        self._orig_modules = {}
        for name in ("ragas", "ragas.metrics", "ragas.llms", "datasets"):
            self._orig_modules[name] = sys.modules.get(name)

        sys.modules["ragas"] = mock_ragas
        sys.modules["ragas.metrics"] = mock_ragas_metrics
        sys.modules["ragas.llms"] = mock_ragas_llms
        sys.modules["datasets"] = mock_datasets

        yield

        # 恢复
        for name, orig in self._orig_modules.items():
            if orig is not None:
                sys.modules[name] = orig
            else:
                sys.modules.pop(name, None)

    @pytest.mark.asyncio
    async def test_returns_eval_metrics_when_ragas_succeeds(self):
        """正常评估返回 EvalMetrics."""
        from ragas import evaluate

        # 构造 evaluate → to_pandas() → iloc[0] → to_dict() mock 链
        mock_row = MagicMock()
        mock_row.to_dict.return_value = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.85,
            "context_recall": 0.95,
        }
        mock_df = MagicMock()
        mock_df.iloc.__getitem__.return_value = mock_row  # df.iloc[0]
        mock_df.__len__.return_value = 1  # len(df) > 0
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = mock_df
        evaluate.return_value = mock_result

        from src.evaluation.metrics import compute_ragas_metrics

        result = await compute_ragas_metrics(
            query="什么是RAG",
            contexts=["RAG是将检索与生成结合的AI架构"],
            answer="RAG是结合检索和生成的AI架构",
            ground_truth="RAG是结合检索和生成的AI架构",
        )

        assert isinstance(result, EvalMetrics)
        assert result.faithfulness == 0.9
        assert result.answer_relevancy == 0.8
        assert result.context_precision == 0.85
        assert result.context_recall == 0.95

    @pytest.mark.asyncio
    async def test_returns_zero_metrics_on_error(self):
        """当 RAGAS 计算抛异常时，返回全零指标而非崩溃."""
        from ragas import evaluate
        evaluate.side_effect = RuntimeError("RAGAS error")

        from src.evaluation.metrics import compute_ragas_metrics

        result = await compute_ragas_metrics(
            query="test",
            contexts=["ctx"],
            answer="ans",
        )

        assert result.faithfulness == 0.0
        assert result.answer_relevancy == 0.0
        assert result.context_precision == 0.0
        assert result.context_recall == 0.0

    @pytest.mark.asyncio
    async def test_handles_missing_ground_truth(self):
        """无 ground_truth 时 context_recall 应为 0.0 (不传入 ground_truth，metrics_to_use 不含 context_recall)."""
        from ragas import evaluate

        # 构造 evaluate → to_pandas() mock 链（返回值不含 context_recall）
        mock_row = MagicMock()
        mock_row.to_dict.return_value = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.85,
        }
        mock_df = MagicMock()
        mock_df.iloc.__getitem__.return_value = mock_row
        mock_df.__len__.return_value = 1
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = mock_df
        evaluate.return_value = mock_result

        from src.evaluation.metrics import compute_ragas_metrics

        result = await compute_ragas_metrics(
            query="query",
            contexts=["ctx"],
            answer="ans",
            ground_truth=None,
        )

        assert result.context_recall == 0.0
        assert result.faithfulness == 0.9  # 其他指标正常

    @pytest.mark.asyncio
    async def test_import_error_raises_runtime_error(self):
        """RAGAS 库不可用（ImportError）时抛出 RuntimeError."""
        from ragas import evaluate
        evaluate.side_effect = ImportError("No module named 'ragas'")

        from src.evaluation.metrics import compute_ragas_metrics

        with pytest.raises(RuntimeError, match="RAGAS library not available"):
            await compute_ragas_metrics("q", ["c"], "a")
