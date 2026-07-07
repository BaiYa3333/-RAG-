"""SSE event type protocol — typed event schemas for all streaming endpoints.

Defines the canonical event types and JSON payloads used by /chat/stream,
/upload/stream, and /evaluate/stream.
"""

from __future__ import annotations

from enum import Enum


class SSEEventType(str, Enum):
    """Canonical SSE event type enumeration."""

    # Chat streaming events
    intent = "intent"
    retrieval_start = "retrieval_start"
    retrieval_chunk = "retrieval_chunk"
    quality_gate = "quality_gate"
    rerank = "rerank"
    token = "token"
    done = "done"
    error = "error"

    # Upload streaming events
    parse_start = "parse_start"
    parse_complete = "parse_complete"
    chunking_start = "chunking_start"
    chunking_complete = "chunking_complete"
    embedding_start = "embedding_start"
    embedding_complete = "embedding_complete"
    ingestion_done = "ingestion_done"

    # Evaluation streaming events
    eval_question_start = "eval_question_start"
    eval_question_complete = "eval_question_complete"
    eval_done = "eval_done"


# ── Chat event payload helpers ────────────────────────────────────


def intent_event(intent: str, confidence: float = 0.0) -> dict:
    return {"intent": intent, "confidence": confidence}


def retrieval_start_event(tier: int) -> dict:
    return {"tier": tier, "status": "started"}


def retrieval_chunk_event(tier: int, doc_count: int, top_scores: list[float] | None = None) -> dict:
    return {"tier": tier, "doc_count": doc_count, "top_scores": top_scores or []}


def quality_gate_event(action: str, tier: int, avg_score: float | None = None, reason: str | None = None) -> dict:
    return {"action": action, "tier": tier, "avg_score": avg_score, "reason": reason}


def rerank_event(method: str, input_count: int, output_count: int) -> dict:
    return {"method": method, "input_count": input_count, "output_count": output_count}


def token_event(token: str) -> dict:
    return {"token": token}


def done_event(
    intent: str = "",
    retrieval_tier: int = 1,
    documents_count: int = 0,
    model: str = "",
    session_id: str | None = None,
    answer: str = "",
    citations: dict | None = None,
) -> dict:
    return {
        "intent": intent,
        "retrieval_tier": retrieval_tier,
        "documents_count": documents_count,
        "model": model,
        "session_id": session_id,
        "answer": answer,
        "citations": citations or {},
    }


def error_event(node: str = "", message: str = "") -> dict:
    return {"node": node, "message": message}


# ── Upload event payload helpers ──────────────────────────────────


def parse_start_event(filename: str) -> dict:
    return {"filename": filename, "status": "parsing"}


def parse_complete_event(filename: str, doc_type: str) -> dict:
    return {"filename": filename, "doc_type": doc_type, "status": "parsed"}


def chunking_start_event(chunk_size: int) -> dict:
    return {"chunk_size": chunk_size, "status": "chunking"}


def chunking_complete_event(chunk_count: int) -> dict:
    return {"chunk_count": chunk_count, "status": "chunked"}


def embedding_start_event(batch_size: int) -> dict:
    return {"batch_size": batch_size, "status": "embedding"}


def embedding_complete_event(count: int) -> dict:
    return {"count": count, "status": "embedded"}


def ingestion_done_event(doc_id: str = "", chunk_count: int = 0, title: str = "") -> dict:
    return {"doc_id": doc_id, "chunk_count": chunk_count, "title": title, "status": "indexed"}


# ── Evaluation event payload helpers ───────────────────────────────


def eval_question_start_event(question_index: int, total_questions: int, question: str) -> dict:
    return {"question_index": question_index, "total_questions": total_questions, "question": question}


def eval_question_complete_event(
    question_index: int,
    faithfulness: float = 0.0,
    answer_relevancy: float = 0.0,
    context_precision: float = 0.0,
    context_recall: float = 0.0,
) -> dict:
    return {
        "question_index": question_index,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }


def eval_done_event(
    faithfulness: float = 0.0,
    answer_relevancy: float = 0.0,
    context_precision: float = 0.0,
    context_recall: float = 0.0,
    avg_score: float = 0.0,
) -> dict:
    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "avg_score": avg_score,
    }
