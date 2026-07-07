"""Typed FastAPI dependency functions for application services.

Each dependency extracts the corresponding service instance from
``request.app.state`` and returns it, or raises HTTP 503 if the
service has not been initialised.
"""

from __future__ import annotations

from fastapi import HTTPException, Request


async def get_memory_service(request: Request):
    """Return the MemoryService instance from app state."""
    service = getattr(request.app.state, "memory_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Memory service not available")
    return service


async def get_kb_service(request: Request):
    """Return the KnowledgeBaseService instance from app state."""
    service = getattr(request.app.state, "kb_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Knowledge base service not available")
    return service


async def get_eval_runner(request: Request):
    """Return the EvaluationRunner instance from app state."""
    runner = getattr(request.app.state, "eval_runner", None)
    if runner is None:
        raise HTTPException(status_code=503, detail="Evaluation runner not available")
    return runner


async def get_indexing_pipeline(request: Request):
    """Return the IndexingPipeline instance from app state."""
    pipeline = getattr(request.app.state, "indexing_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Indexing pipeline not available")
    return pipeline
