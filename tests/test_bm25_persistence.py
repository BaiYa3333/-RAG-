"""Unit tests for BM25 persistence — Task 8.3."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest


class TestBM25Persistence:
    """Tests for BM25 save/load, incremental update, atomic write, corrupted file fallback."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def sample_docs(self):
        return [
            {"id": "d1", "content": "微服务架构是一种将应用拆分为小型独立服务的架构风格", "metadata": {}},
            {"id": "d2", "content": "RRF 是 Reciprocal Rank Fusion 的缩写，用于融合多个检索结果", "metadata": {}},
            {"id": "d3", "content": "BM25 是一种基于概率检索模型的排序函数", "metadata": {}},
        ]

    def test_save_and_load_roundtrip(self, temp_dir, sample_docs):
        """保存后加载应恢复文档、倒排索引、IDF 和所有统计信息."""
        from src.rag.retrieval.sparse import SparseRetriever, _index_path

        with patch("src.rag.retrieval.sparse._index_dir", return_value=temp_dir):
            sp = SparseRetriever(collection_name="test_kb")
            sp.build_index(sample_docs)

            assert sp.save_index("test_kb") is True

            # 验证文件存在
            path = _index_path("test_kb")
            assert os.path.exists(path)

            # 加载
            sp2 = SparseRetriever(collection_name="test_kb")
            assert sp2.load_index("test_kb") is True
            # 元数据
            assert sp2._avg_doc_length == sp._avg_doc_length
            assert sp2._document_frequency == sp._document_frequency
            assert sp2._doc_lengths == sp._doc_lengths
            # 文档已恢复
            assert len(sp2._documents) == len(sample_docs)
            assert sp2._documents[0]["content"] == sample_docs[0]["content"]
            # 倒排索引已重建
            assert len(sp2._inverted_index) > 0
            # 可以直接检索（无需再调用 build_index）
            results = sp2.search("微服务架构", top_k=2)
            assert len(results) > 0
            assert any("微服务" in r["content"] for r in results)

    def test_atomic_write(self, temp_dir, sample_docs):
        """原子写入：先写 .tmp，完成后再 os.replace."""
        from src.rag.retrieval.sparse import SparseRetriever, _index_path, _tmp_path

        with patch("src.rag.retrieval.sparse._index_dir", return_value=temp_dir):
            sp = SparseRetriever(collection_name="test_kb")
            sp.build_index(sample_docs)

            # 删除可能的残留文件
            tmp = _tmp_path("test_kb")
            if os.path.exists(tmp):
                os.remove(tmp)

            assert sp.save_index("test_kb") is True

            # .tmp 文件应已被 rename，不应存在
            assert not os.path.exists(tmp)

            # 最终 JSON 文件应存在
            path = _index_path("test_kb")
            assert os.path.exists(path)

            # 验证 JSON 内容完整性
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["collection"] == "test_kb"
            assert data["num_docs"] == 3
            assert "document_frequency" in data
            assert "doc_lengths" in data
            assert "documents" in data
            assert len(data["documents"]) == 3
            assert "updated_at" in data

    def test_incremental_update(self, temp_dir, sample_docs):
        """增量添加文档应合并词项统计."""
        from src.rag.retrieval.sparse import SparseRetriever

        with patch("src.rag.retrieval.sparse._index_dir", return_value=temp_dir):
            sp = SparseRetriever(collection_name="test_kb")
            sp.build_index(sample_docs[:2])  # 初始 2 个文档

            old_df_sum = sum(sp._document_frequency.values())
            old_count = len(sp._documents)

            # 增量添加第 3 个文档
            new_docs = [
                {"id": "d3", "content": "BM25 是一种基于概率检索模型的排序函数", "metadata": {}}
            ]
            sp.add_documents(new_docs)

            assert len(sp._documents) == 3
            assert len(sp._doc_lengths) == 3
            # DF 应增加
            new_df_sum = sum(sp._document_frequency.values())
            assert new_df_sum >= old_df_sum

    def test_corrupted_file_fallback(self, temp_dir, sample_docs):
        """损坏的 JSON 文件应返回 False（触发全量重建）."""
        from src.rag.retrieval.sparse import SparseRetriever, _index_path

        path = _index_path("test_kb")
        real_dir = os.path.dirname(path)

        with patch("src.rag.retrieval.sparse._index_dir", return_value=temp_dir):
            # 写入损坏的 JSON
            os.makedirs(real_dir, exist_ok=True)
            with open(path, "w") as f:
                f.write("this is not valid json {{{")

            sp = SparseRetriever(collection_name="test_kb")
            assert sp.load_index("test_kb") is False

    def test_missing_file_returns_false(self, temp_dir):
        """不存在的 JSON 文件应返回 False."""
        from src.rag.retrieval.sparse import SparseRetriever

        with patch("src.rag.retrieval.sparse._index_dir", return_value=temp_dir):
            sp = SparseRetriever(collection_name="nonexistent")
            assert sp.load_index("nonexistent") is False

    def test_save_no_collection_name(self, sample_docs):
        """无 collection_name 时 save_index 应返回 False."""
        from src.rag.retrieval.sparse import SparseRetriever

        sp = SparseRetriever()  # No collection_name
        sp.build_index(sample_docs)
        assert sp.save_index() is False

    def test_search_after_build(self, sample_docs):
        """构建索引后应能正常检索."""
        from src.rag.retrieval.sparse import SparseRetriever

        sp = SparseRetriever()
        sp.build_index(sample_docs)

        results = sp.search("微服务架构", top_k=2)
        assert len(results) > 0
        assert any("微服务" in r["content"] for r in results)

    def test_search_after_load_from_disk(self, temp_dir, sample_docs):
        """从磁盘加载后应能直接检索，无需再调用 build_index."""
        from src.rag.retrieval.sparse import SparseRetriever

        with patch("src.rag.retrieval.sparse._index_dir", return_value=temp_dir):
            # 创建、构建、保存
            sp = SparseRetriever(collection_name="test_kb")
            sp.build_index(sample_docs)
            sp.save_index("test_kb")

            # 全新实例：不调用 build_index，直接 search
            sp2 = SparseRetriever(collection_name="test_kb")
            results = sp2.search("微服务架构", top_k=2)

            # 应成功从磁盘懒加载并正常检索
            assert len(results) > 0
            assert any("微服务" in r["content"] for r in results)
            # 确认倒排索引已重建
            assert len(sp2._inverted_index) > 0
            # 确认标记为已从磁盘加载
            assert sp2._loaded_from_disk is True

    def test_multiple_search_calls_no_silent_fail(self, temp_dir, sample_docs):
        """多次 search() 调用应持续返回结果，不应静默返回空列表（回归测试）."""
        from src.rag.retrieval.sparse import SparseRetriever

        with patch("src.rag.retrieval.sparse._index_dir", return_value=temp_dir):
            # 创建、构建、保存
            sp = SparseRetriever(collection_name="test_kb")
            sp.build_index(sample_docs)
            sp.save_index("test_kb")

            # 全新实例：多次调用 search
            sp2 = SparseRetriever(collection_name="test_kb")

            # 第一次调用：触发懒加载
            r1 = sp2.search("微服务", top_k=2)
            assert len(r1) > 0, "第一次 search 不应返回空"

            # 第二次调用：不应静默返回 []
            r2 = sp2.search("BM25", top_k=2)
            assert len(r2) > 0, "第二次 search 不应返回空（回归：_loaded_from_disk 已为 True 但倒排索引应存在）"

            # 第三次调用：同样应有结果
            r3 = sp2.search("RRF", top_k=2)
            assert len(r3) > 0, "第三次 search 不应返回空"
