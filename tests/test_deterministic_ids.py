"""测试确定性 Chunk ID — 可复现性、内容变更触发 ID 变化."""

import hashlib
import uuid
from unittest.mock import patch

import pytest

from src.config import settings
from src.rag.indexing.chunker import (
    ParentChildChunker,
    _compute_content_hash,
    _generate_deterministic_id,
    _generate_deterministic_parent_id,
)


# ── _compute_content_hash ──────────────────────────

def test_compute_content_hash():
    h = _compute_content_hash("hello")
    assert len(h) == 8
    assert h == hashlib.sha256("hello".encode("utf-8")).hexdigest()[:8]


def test_compute_content_hash_deterministic():
    """相同内容产生相同 hash。"""
    h1 = _compute_content_hash("same content")
    h2 = _compute_content_hash("same content")
    assert h1 == h2


def test_compute_content_hash_different():
    """不同内容产生不同 hash。"""
    h1 = _compute_content_hash("content A")
    h2 = _compute_content_hash("content B")
    assert h1 != h2


# ── _generate_deterministic_id ─────────────────────

def test_generate_deterministic_id_format():
    doc_hash = "a1b2c3d4e5f6" + "0" * 52  # 64 chars
    chunk_id = _generate_deterministic_id(doc_hash, 0, 3, "9f8e7d6c")
    assert chunk_id == "a1b2c3d4e5f6_p0000_c0003_9f8e7d6c"


def test_generate_deterministic_parent_id_format():
    doc_hash = "a1b2c3d4e5f6" + "0" * 52
    parent_id = _generate_deterministic_parent_id(doc_hash, 5)
    assert parent_id == "a1b2c3d4e5f6_p0005"


# ── Deterministic IDs (integration) ────────────────

def test_chunker_without_doc_hash_uses_uuid():
    """没有 doc_hash 且确定性模式也应 fallback 到 UUID？不，它会尝试用 content hash。"""
    # 实际上 settings.ingestion_deterministic_ids=True 时，会从 content 推导 doc_hash
    chunker = ParentChildChunker(child_size=100, parent_size=500)
    docs = [{"content": "测试内容 A", "metadata": {"source": "test.txt"}}]
    chunks = chunker.split_documents(docs)
    assert len(chunks) > 0
    # 确定性 IDs 应包含 doc_hash 前缀
    assert "_p" in chunks[0]["chunk_id"]
    assert "_c" in chunks[0]["chunk_id"]


def test_chunker_id_reproducibility():
    """相同文档两次摄入应产生相同 ID。"""
    chunker = ParentChildChunker(child_size=200, parent_size=500)
    docs = [{"content": "这是测试内容，用于验证 ID 的确定性生成。需要足够长以被 split。", "metadata": {"source": "test.txt"}}]

    chunks1 = chunker.split_documents(docs, doc_hash="a1b2c3d4e5f6" + "0" * 52)
    chunks2 = chunker.split_documents(docs, doc_hash="a1b2c3d4e5f6" + "0" * 52)

    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1["chunk_id"] == c2["chunk_id"]


def test_chunker_content_change_triggers_new_id():
    """内容变更应产生不同的 chunk_id。"""
    chunker = ParentChildChunker(child_size=200, parent_size=500)
    docs1 = [{"content": "版本A: 原始内容", "metadata": {"source": "test.txt"}}]
    docs2 = [{"content": "版本B: 修改后的内容", "metadata": {"source": "test.txt"}}]

    chunks1 = chunker.split_documents(docs1, doc_hash="a1b2c3d4e5f6" + "0" * 52)
    chunks2 = chunker.split_documents(docs2, doc_hash="a1b2c3d4e5f6" + "0" * 52)

    # content_hash 应不同
    ids1 = {c["chunk_id"] for c in chunks1}
    ids2 = {c["chunk_id"] for c in chunks2}
    assert ids1 != ids2


def test_chunker_metadata_includes_hash_fields():
    """Chunk metadata 应包含 content_hash 和 doc_hash。"""
    chunker = ParentChildChunker(child_size=100, parent_size=500)
    docs = [{"content": "测试内容 A", "metadata": {"source": "test.txt"}}]
    chunks = chunker.split_documents(docs, doc_hash="a1b2c3d4e5f6" + "0" * 52)
    for c in chunks:
        assert "content_hash" in c["metadata"]
        assert "doc_hash" in c["metadata"]
        assert len(c["metadata"]["content_hash"]) == 8
