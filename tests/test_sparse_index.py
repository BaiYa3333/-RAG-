"""Test SparseRetriever — inverted index BM25 with JSON persistence."""

import json
import os
import tempfile
import pytest

import jieba


# Override the index directory for testing
@pytest.fixture(autouse=True)
def _temp_index_dir(monkeypatch):
    """Redirect BM25 index dir to a temp directory for isolation."""
    tmp = tempfile.mkdtemp(prefix="bm25_test_")
    monkeypatch.setattr(
        "src.rag.retrieval.sparse._index_dir",
        lambda: tmp,
    )
    monkeypatch.setattr(
        "src.config.settings.rag_bm25_index_dir",
        tmp,
    )
    yield tmp
    # Cleanup
    import shutil
    try:
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def sample_docs():
    return [
        {"id": "doc_0", "content": "混合检索通过RRF算法融合稠密检索和稀疏检索结果", "metadata": {"type": "tech"}},
        {"id": "doc_1", "content": "员工门诊甲类药品报销90%，乙类药品报销80%", "metadata": {"type": "health"}},
        {"id": "doc_2", "content": "Embedding模型使用text-embedding-v4生成1024维向量", "metadata": {"type": "tech"}},
        {"id": "doc_3", "content": "意图路由器将查询分为事实、比较、总结、分析四种", "metadata": {"type": "tech"}},
        {"id": "doc_4", "content": "稠密检索基于向量语义相似度匹配，稀疏检索使用BM25关键词检索", "metadata": {"type": "tech"}},
    ]


@pytest.fixture
def retriever():
    from src.rag.retrieval.sparse import SparseRetriever
    return SparseRetriever(collection_name="test_collection")


class TestInvertedIndexBuild:
    """Test inverted index construction."""

    def test_build_index_populates_structures(self, retriever, sample_docs):
        retriever.build_index(sample_docs)

        assert len(retriever._documents) == 5
        assert len(retriever._doc_lengths) == 5
        assert retriever._avg_doc_length > 0
        assert len(retriever._inverted_index) > 0
        assert len(retriever._document_frequency) > 0
        assert len(retriever._idf) > 0
        assert retriever._idf.keys() == retriever._document_frequency.keys()

    def test_inverted_index_structure(self, retriever, sample_docs):
        retriever.build_index(sample_docs)

        # 每个词项的 posting 中 doc_idx 应对，tf 为正整数
        for term, postings in retriever._inverted_index.items():
            assert isinstance(postings, dict)
            for doc_idx, tf in postings.items():
                assert isinstance(doc_idx, int)
                assert isinstance(tf, int)
                assert tf > 0
                assert doc_idx < len(sample_docs)

    def test_document_frequency_consistent(self, retriever, sample_docs):
        retriever.build_index(sample_docs)

        num_docs = len(sample_docs)
        for term, df in retriever._document_frequency.items():
            assert 1 <= df <= num_docs
            # inverted_index 中该 term 的 posting 数量应等于 df
            assert len(retriever._inverted_index[term]) == df

    def test_avg_doc_length_correct(self, retriever, sample_docs):
        retriever.build_index(sample_docs)

        expected_avg = sum(retriever._doc_lengths) / len(retriever._doc_lengths)
        assert retriever._avg_doc_length == pytest.approx(expected_avg)


class TestBM25Scoring:
    """Test manual BM25 scoring."""

    def test_relevant_docs_rank_higher(self, retriever, sample_docs):
        retriever.build_index(sample_docs)

        # Query about retrieval → docs with "检索" should rank higher
        results = retriever.search("检索策略", top_k=5)
        assert len(results) > 0

        # doc_0 and doc_4 mention retrieval-related terms
        top_ids = [r["id"] for r in results]
        # The retrieval-related docs should be near the top
        retrieval_ids = {"doc_0", "doc_4"}
        assert any(did in retrieval_ids for did in top_ids[:3])

    def test_health_query_ranks_health_doc(self, retriever, sample_docs):
        retriever.build_index(sample_docs)

        results = retriever.search("药品报销", top_k=3)
        assert len(results) > 0
        # doc_1 is the health document
        assert results[0]["id"] == "doc_1" or results[1]["id"] == "doc_1"

    def test_scores_are_positive(self, retriever, sample_docs):
        retriever.build_index(sample_docs)

        results = retriever.search("混合检索融合", top_k=5)
        for r in results:
            assert r["score"] > 0

    def test_empty_query_returns_empty(self, retriever, sample_docs):
        retriever.build_index(sample_docs)

        results = retriever.search("", top_k=5)
        # Empty query after tokenization produces no tokens → all scores 0
        # Filtered out by score > 0
        assert len(results) == 0


class TestJSONPersistence:
    """Test save / load round-trip."""

    def test_save_and_load_roundtrip(self, retriever, sample_docs, _temp_index_dir):
        retriever.build_index(sample_docs)
        assert retriever.save_index()

        # Create a fresh retriever and load
        from src.rag.retrieval.sparse import SparseRetriever
        retriever2 = SparseRetriever(collection_name="test_collection")
        assert retriever2.load_index()

        # Verify structures
        assert len(retriever2._documents) == len(sample_docs)
        assert retriever2._doc_lengths == retriever._doc_lengths
        assert retriever2._avg_doc_length == pytest.approx(retriever._avg_doc_length)
        assert retriever2._document_frequency == retriever._document_frequency
        assert retriever2._inverted_index == retriever._inverted_index
        assert retriever2._idf == retriever._idf

        # Search should produce same results
        r1 = retriever.search("检索策略", top_k=3)
        r2 = retriever2.search("检索策略", top_k=3)
        assert [d["id"] for d in r1] == [d["id"] for d in r2]
        for a, b in zip(r1, r2):
            assert a["score"] == pytest.approx(b["score"])

    def test_load_nonexistent_file(self, retriever):
        retriever2 = retriever
        from src.rag.retrieval.sparse import SparseRetriever
        r = SparseRetriever(collection_name="nonexistent")
        assert r.load_index() is False

    def test_saved_json_contains_required_fields(self, retriever, sample_docs, _temp_index_dir):
        retriever.build_index(sample_docs)
        retriever.save_index()

        # Read the JSON file
        from src.rag.retrieval.sparse import _index_path
        path = _index_path("test_collection")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "collection" in data
        assert "num_docs" in data
        assert "avg_doc_length" in data
        assert "doc_lengths" in data
        assert "document_frequency" in data
        assert "inverted_index" in data
        assert "idf" in data
        assert "documents" in data
        assert "updated_at" in data

        # 验证倒排索引 JSON 序列化正确（doc_idx 为 str key）
        for term, postings in data["inverted_index"].items():
            assert isinstance(postings, dict)
            for doc_idx_str, tf in postings.items():
                assert doc_idx_str.isdigit() or doc_idx_str.lstrip("-").isdigit()
                assert isinstance(tf, int)


class TestIncrementalUpdate:
    """Test add_documents for incremental index updates."""

    def test_add_documents_increases_size(self, retriever, sample_docs):
        retriever.build_index(sample_docs[:3])
        old_count = len(retriever._documents)

        retriever.add_documents(sample_docs[3:])
        assert len(retriever._documents) == old_count + 2

    def test_idf_updated_after_addition(self, retriever, sample_docs):
        retriever.build_index(sample_docs[:3])
        old_idf = dict(retriever._idf)

        retriever.add_documents(sample_docs[3:])

        # IDF values should be recalculated (may differ due to N change)
        assert retriever._idf != old_idf or len(retriever._idf) > len(old_idf)

    def test_avg_doc_length_updated(self, retriever, sample_docs):
        retriever.build_index(sample_docs[:3])
        old_avg = retriever._avg_doc_length

        retriever.add_documents(sample_docs[3:])

        # avg_doc_length should change (unless by coincidence)
        expected_avg = sum(retriever._doc_lengths) / len(retriever._doc_lengths)
        assert retriever._avg_doc_length == pytest.approx(expected_avg)

    def test_search_works_after_addition(self, retriever, sample_docs):
        retriever.build_index(sample_docs[:3])
        retriever.add_documents(sample_docs[3:])

        results = retriever.search("检索", top_k=5)
        assert len(results) > 0

    def test_add_empty_docs_noop(self, retriever, sample_docs):
        retriever.build_index(sample_docs)
        old_docs = len(retriever._documents)
        retriever.add_documents([])
        assert len(retriever._documents) == old_docs


class TestBackwardCompatibility:
    """Test loading old-format JSON (without inverted_index / idf)."""

    def test_load_legacy_format(self, retriever, sample_docs, _temp_index_dir):
        """Simulate old JSON format: documents + document_frequency only."""
        from src.rag.retrieval.sparse import _index_path
        path = _index_path("test_collection")

        # Write a legacy-format JSON (no inverted_index, no idf)
        legacy_data = {
            "collection": "test_collection",
            "num_docs": len(sample_docs),
            "avg_doc_length": 10.0,
            "doc_lengths": [10, 12, 14, 9, 11],
            "document_frequency": {"检索": 3, "混合": 1, "药品": 1},
            "documents": sample_docs,
            "updated_at": "2026-01-01T00:00:00Z",
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(legacy_data, f, ensure_ascii=False, indent=2)

        from src.rag.retrieval.sparse import SparseRetriever
        r = SparseRetriever(collection_name="test_collection")
        assert r.load_index() is True

        # 应该从 document_frequency 重建了 inverted_index 和 idf
        assert len(r._inverted_index) > 0
        assert len(r._idf) > 0
        # 搜索应正常工作
        results = r.search("检索", top_k=3)
        assert len(results) > 0


class TestLargeCorpus:
    """Test index with larger corpus — stress test."""

    def test_many_documents(self, retriever):
        """Build index with many documents to ensure no performance issues."""
        docs = [
            {"id": f"doc_{i}", "content": f"这是第{i}个测试文档，包含检索、排序、BM25、关键词匹配等内容",
             "metadata": {}}
            for i in range(500)
        ]
        retriever.build_index(docs)

        assert len(retriever._documents) == 500
        assert len(retriever._inverted_index) > 0
        assert len(retriever._idf) > 0

        results = retriever.search("BM25关键词检索", top_k=10)
        assert len(results) > 0
        assert len(results) <= 10
