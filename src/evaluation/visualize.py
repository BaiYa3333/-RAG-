"""RAGAS 评估可视化 — Plotly 雷达图 + 趋势折线图 + 明细表 + Dashboard HTML 组装."""

from __future__ import annotations

# 颜色系统 — 与 spec 定义一致
COLORS = {
    "faithfulness": "#2196F3",       # 蓝
    "answer_relevancy": "#4CAF50",   # 绿
    "context_precision": "#FF9800",  # 橙
    "context_recall": "#9C27B0",    # 紫
    "avg_score": "#F44336",          # 红（虚线参考）
}

METRIC_LABELS_CN = {
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关性",
    "context_precision": "上下文精度",
    "context_recall": "上下文召回率",
    "avg_score": "平均分",
}

METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def generate_radar_chart(metrics: dict) -> "plotly.graph_objects.Figure":
    """生成 Plotly 极坐标雷达图，展示单次评估的 4 项指标。

    Args:
        metrics: {"faithfulness": 0.85, "answer_relevancy": 0.78, ...}

    Returns:
        go.Figure — 雷达图对象
    """
    import plotly.graph_objects as go

    values = [metrics.get(k, 0.0) for k in METRIC_KEYS]
    # 闭合雷达图
    values_closed = values + [values[0]]
    labels_cn = [METRIC_LABELS_CN[k] for k in METRIC_KEYS]
    labels_closed = labels_cn + [labels_cn[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(33, 150, 243, 0.25)",
        line=dict(color=COLORS["faithfulness"], width=2),
        name="当前评估",
        hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                gridcolor="rgba(0,0,0,0.1)",
            ),
        ),
        showlegend=True,
        margin=dict(l=60, r=60, t=40, b=40),
        title=dict(
            text="RAGAS 评估雷达图",
            x=0.5,
            font=dict(size=16),
        ),
    )

    return fig


def generate_trend_chart(history: list[dict]) -> "plotly.graph_objects.Figure":
    """生成 Plotly 趋势折线图，展示多轮评估的指标变化。

    Args:
        history: 评估历史列表（含 created_at 和 4 项指标）

    Returns:
        go.Figure — 趋势图对象
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    if len(history) < 2:
        # 单次评估：显示单点标注
        if history:
            item = history[0]
            x_vals = [item.get("created_at", "")[:10]]
            fig.add_annotation(
                text="需要至少 2 次评估才能显示趋势",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#888"),
            )
            for key in METRIC_KEYS:
                if key in item:
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=[item[key]],
                        mode="markers+text",
                        name=METRIC_LABELS_CN[key],
                        line=dict(color=COLORS[key], width=2),
                        marker=dict(size=10),
                    ))
        else:
            fig.add_annotation(
                text="暂无评估数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#888"),
            )
    else:
        # 多轮评估：画折线图
        x_vals = [h.get("created_at", "")[:10] for h in history]

        for key in METRIC_KEYS:
            y_vals = [h.get(key, 0.0) for h in history]
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=METRIC_LABELS_CN[key],
                line=dict(color=COLORS[key], width=2),
                marker=dict(size=6),
                hovertemplate="%{x}<br>%{y:.3f}<extra>%{fullData.name}</extra>",
            ))

        # avg_score 虚线参考线
        avg_vals = [h.get("avg_score", 0.0) for h in history]
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=avg_vals,
            mode="lines",
            name="平均分",
            line=dict(color=COLORS["avg_score"], width=1.5, dash="dash"),
            hovertemplate="%{x}<br>平均: %{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        yaxis=dict(
            title="分数",
            range=[0, 1],
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            gridcolor="rgba(0,0,0,0.08)",
        ),
        xaxis=dict(
            title="评估时间",
            gridcolor="rgba(0,0,0,0.05)",
        ),
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=50, r=30, t=40, b=80),
        title=dict(
            text="评估指标趋势",
            x=0.5,
            font=dict(size=16),
        ),
    )

    return fig


def generate_detail_table(history: list[dict]) -> "plotly.graph_objects.Figure":
    """生成 Plotly 明细表，展示评估历史的完整数据。

    Args:
        history: 评估历史列表

    Returns:
        go.Figure — 表格对象
    """
    import plotly.graph_objects as go

    if not history:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无评估数据",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#888"),
        )
        return fig

    # 按时间降序排列，取最近 20 条
    sorted_history = sorted(history, key=lambda h: h.get("created_at", ""), reverse=True)[:20]

    header_values = ["评估 ID", "时间", "忠实度", "答案相关性", "上下文精度", "上下文召回率", "平均分"]
    cells = {
        "eval_id": [],
        "time": [],
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
        "avg_score": [],
    }

    for h in sorted_history:
        cells["eval_id"].append(h.get("evaluation_id", "")[:8])
        cells["time"].append((h.get("created_at", "") or "")[:19])
        cells["faithfulness"].append(f'{h.get("faithfulness", 0):.3f}')
        cells["answer_relevancy"].append(f'{h.get("answer_relevancy", 0):.3f}')
        cells["context_precision"].append(f'{h.get("context_precision", 0):.3f}')
        cells["context_recall"].append(f'{h.get("context_recall", 0):.3f}')
        cells["avg_score"].append(f'{h.get("avg_score", 0):.3f}')

    # 颜色编码：高分绿色，中分黄色，低分红色
    cell_colors: list[list[str]] = [[], [], [], [], [], [], []]
    color_indices = [2, 3, 4, 5, 6]  # 指标列索引

    for i, h in enumerate(sorted_history):
        for j in range(7):
            if j in color_indices:
                val = float(list(cells.values())[j][i]) if list(cells.values())[j][i] else 0.0
                if val >= 0.8:
                    cell_colors[j].append("rgba(76, 175, 80, 0.15)")  # 绿色浅底
                elif val >= 0.6:
                    cell_colors[j].append("rgba(255, 152, 0, 0.12)")  # 橙色浅底
                else:
                    cell_colors[j].append("rgba(244, 67, 54, 0.12)")  # 红色浅底
            else:
                cell_colors[j].append("white")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_values,
            fill_color="rgba(33, 150, 243, 0.15)",
            align="center",
            font=dict(size=12),
            height=35,
        ),
        cells=dict(
            values=list(cells.values()),
            fill_color=cell_colors,
            align="center",
            font=dict(size=11),
            height=30,
        ),
    )])

    fig.update_layout(
        title=dict(
            text="评估明细表（最近 20 次）",
            x=0.5,
            font=dict(size=16),
        ),
        margin=dict(l=10, r=10, t=50, b=20),
    )

    return fig


def build_dashboard_html(
    radar_fig,
    table_fig,
    last_updated: str = "",
) -> str:
    """组装两个 Plotly 图表为单个自包含 HTML Dashboard 页面。

    Args:
        radar_fig: 雷达图 go.Figure
        table_fig: 明细表 go.Figure
        last_updated: 最后更新时间字符串

    Returns:
        完整 HTML 字符串
    """
    import plotly

    radar_html = plotly.io.to_html(radar_fig, full_html=False, include_plotlyjs="cdn")
    table_html = plotly.io.to_html(table_fig, full_html=False, include_plotlyjs=False)

    subtitle = f"最后更新: {last_updated}" if last_updated else "暂无数据"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAGAS 评估 Dashboard</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #f5f7fa;
    color: #333;
    padding: 20px;
}}
.header {{
    text-align: center;
    padding: 20px 0 10px;
}}
.header h1 {{
    font-size: 24px;
    color: #1976D2;
    margin-bottom: 6px;
}}
.header p {{
    font-size: 13px;
    color: #888;
}}
.dashboard-grid {{
    display: flex;
    flex-direction: column;
    gap: 20px;
    max-width: 900px;
    margin: 0 auto;
}}
.chart-card {{
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    padding: 16px;
    overflow: hidden;
}}
</style>
</head>
<body>

<div class="header">
    <h1>📊 RAGAS 评估 Dashboard</h1>
    <p>{subtitle}</p>
</div>

<div class="dashboard-grid">
    <div class="chart-card">
        {radar_html}
    </div>
    <div class="chart-card">
        {table_html}
    </div>
</div>

</body>
</html>"""
