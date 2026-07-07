"""API 错误处理 — RAGException → HTTP 状态码映射."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from src.utils.exceptions import (
    RAGException,
    IngestionQualityError,
    RetrievalError,
    GenerationError,
    LLMError,
)
from src.utils.logger import logger


async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    """将 RAGException 层次映射到 HTTP 状态码。

    RetrievalError / LLMError → 503 (服务不可用，可重试)
    GenerationError         → 503 (生成失败)
    其他 RAGException       → 500
    """
    status = 500
    if isinstance(exc, IngestionQualityError):
        status = 422
    elif isinstance(exc, (RetrievalError, LLMError, GenerationError)):
        status = 503

    logger.error(
        "api_exception",
        path=request.url.path,
        exception_type=type(exc).__name__,
        detail=str(exc),
        status=status,
    )
    return JSONResponse(
        status_code=status,
        content={"detail": str(exc)},
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """未预期的异常 → 500."""
    logger.error(
        "api_unhandled_exception",
        path=request.url.path,
        exception_type=type(exc).__name__,
        detail=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
