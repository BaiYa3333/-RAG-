"""RAG_Project 结构化日志 — structlog + JSON 输出.

setup_logger() 可在 lifespan 中显式调用（使用配置的 log_level），
也可在测试 conftest 中覆盖。不设置 cache_logger_on_first_use，
允许重复配置。
"""

import logging
import os
import structlog

_configured = False


def setup_logger(log_level: str = "INFO", stream=None) -> structlog.stdlib.BoundLogger:
    """配置 structlog，返回绑定了日志级别的 logger。

    可安全重复调用 — 后续调用会覆盖之前的配置（测试友好）。

    渲染器选择基于 ENV 环境变量：
    - ENV=development（默认）→ ConsoleRenderer（彩色控制台输出）
    - ENV=production → JSONRenderer（机器可解析的 JSON 行）

    Args:
        log_level: 日志级别名。
        stream: 日志输出流，默认 None（stdout）。MCP stdio 模式必须传
            sys.stderr — stdout 是 JSON-RPC 通道，日志写入会破坏协议帧。
    """
    global _configured
    level = getattr(logging, log_level.upper(), logging.INFO)

    env = os.getenv("ENV", "development")
    renderer = (
        structlog.dev.ConsoleRenderer()
        if env == "development"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=stream),
        cache_logger_on_first_use=False,
    )

    _configured = True
    return structlog.get_logger()


def get_logger() -> structlog.stdlib.BoundLogger:
    """获取当前 structlog logger（若未配置则使用默认初始化）。"""
    global _configured
    if not _configured:
        setup_logger()
    return structlog.get_logger()


# 模块级 logger 实例（向后兼容），首次导入时用默认配置初始化
logger = get_logger()
