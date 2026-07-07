"""测试 SHA256 文件完整性校验 (IngestionCache)。"""

import os
import tempfile

import pytest

from src.rag.ingestion.integrity import IngestionCache


@pytest.fixture
def cache():
    """使用临时数据库创建 IngestionCache。"""
    import gc
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    cache = IngestionCache(db_path=tmp.name)
    yield cache
    # 清理连接以允许 Windows 删除文件
    del cache
    gc.collect()
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass  # Windows 偶发文件锁，跳过清理


@pytest.fixture
def tmp_file():
    """创建临时文件用于测试。"""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp.write("test content for ingestion integrity check")
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


def test_should_skip_new_file(cache, tmp_file):
    """新文件不应被跳过。"""
    import hashlib
    file_hash = hashlib.sha256(b"test content for ingestion integrity check").hexdigest()
    assert cache.should_skip(file_hash) is False


def test_mark_success_and_skip(cache, tmp_file):
    """成功后应被 skip。"""
    import hashlib
    file_hash = hashlib.sha256(b"test content for ingestion integrity check").hexdigest()
    cache.mark_success(file_hash, tmp_file, collection="test_kb", chunk_count=5)
    assert cache.should_skip(file_hash) is True


def test_force_reprocess(cache, tmp_file):
    """即使已成功，force 模式也应允许重处理（由 pipeline 层控制）。"""
    import hashlib
    file_hash = hashlib.sha256(b"test content for ingestion integrity check").hexdigest()
    cache.mark_success(file_hash, tmp_file, collection="test_kb", chunk_count=5)
    # should_skip 返回 True 表示"可以跳过"，但 force 模式在 pipeline 层忽略此结果
    assert cache.should_skip(file_hash) is True


def test_mark_failed(cache, tmp_file):
    """失败记录不应被 skip。"""
    import hashlib
    file_hash = hashlib.sha256(b"test content for ingestion integrity check").hexdigest()
    cache.mark_failed(file_hash, tmp_file, "test error")
    assert cache.should_skip(file_hash) is False


def test_list_processed(cache, tmp_file):
    """list_processed 应返回记录。"""
    import hashlib
    file_hash = hashlib.sha256(b"test content for ingestion integrity check").hexdigest()
    cache.mark_success(file_hash, tmp_file, collection="test_kb", chunk_count=10)
    results = cache.list_processed()
    assert len(results) >= 1
    assert results[0]["file_hash"] == file_hash
    assert results[0]["status"] == "success"
    assert results[0]["chunk_count"] == 10


def test_list_processed_by_collection(cache, tmp_file):
    """按 collection 过滤 list_processed。"""
    import hashlib
    file_hash = hashlib.sha256(b"test content for ingestion integrity check").hexdigest()
    cache.mark_success(file_hash, tmp_file, collection="kb_a", chunk_count=3)
    results_a = cache.list_processed(collection="kb_a")
    results_b = cache.list_processed(collection="kb_b")
    assert len(results_a) >= 1
    assert len(results_b) == 0
