"""RAGAS 评估模块 — 指标计算、测试集生成、评估流程编排、可视化."""

from src.evaluation.metrics import compute_ragas_metrics
from src.evaluation.testset_generator import generate_testset
from src.evaluation.history import (
    save_evaluation_result,
    get_evaluation_results,
    get_latest_evaluation,
)
from src.evaluation.ragas_runner import EvaluationRunner
from src.evaluation.visualize import (
    generate_radar_chart,
    generate_detail_table,
    build_dashboard_html,
)

__all__ = [
    "compute_ragas_metrics",
    "generate_testset",
    "save_evaluation_result",
    "get_evaluation_results",
    "get_latest_evaluation",
    "EvaluationRunner",
    "generate_radar_chart",
    "generate_detail_table",
    "build_dashboard_html",
]
