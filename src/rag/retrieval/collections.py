"""Shared ChromaDB collection naming helpers."""

from __future__ import annotations

import hashlib
import re

_COLLECTION_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]")
_MAX_COLLECTION_NAME_LEN = 63
_MIN_COLLECTION_NAME_LEN = 3


def _sanitize_collection_part(value: str) -> str:
    cleaned = _COLLECTION_SAFE_RE.sub("_", value).strip("._-")
    cleaned = re.sub(r"_{2,}", "_", cleaned)
    return cleaned or "id"


def _bounded_collection_name(prefix: str, raw_id: str) -> str:
    safe = _sanitize_collection_part(raw_id)
    name = f"{prefix}{safe}"
    if len(name) <= _MAX_COLLECTION_NAME_LEN:
        return name if len(name) >= _MIN_COLLECTION_NAME_LEN else f"{name}_x"

    digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
    keep = _MAX_COLLECTION_NAME_LEN - len(prefix) - len(digest) - 1
    shortened = safe[:max(1, keep)].rstrip("._-") or "id"
    return f"{prefix}{shortened}_{digest}"


def session_collection_name(session_id: str) -> str:
    return _bounded_collection_name("rag_session_", session_id)


def kb_collection_name(kb_id: str) -> str:
    return _bounded_collection_name("rag_kb_", kb_id)
