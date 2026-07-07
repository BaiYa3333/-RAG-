"""Observability module — Langfuse trace/span monitoring for RAG pipeline.

Exports:
    get_langfuse: Obtain the global Langfuse client singleton (or None if disabled).
    flush_langfuse: Flush pending trace data to Langfuse Server.
    trace_rag_node: Decorator for creating node-level spans.
    track_llm_call: Helper to record LLM generation spans with token usage.
"""

from src.observability.client import get_langfuse, flush_langfuse
from src.observability.decorators import trace_rag_node

__all__ = [
    "get_langfuse",
    "flush_langfuse",
    "trace_rag_node",
]
