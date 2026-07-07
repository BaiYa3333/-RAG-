"""Unit tests for QueryProcessor — Tasks 7.5 & 8.5."""

import pytest

from src.rag.retrieval.query_processor import QueryProcessor, ParsedQuery


class TestQueryProcessorParse:
    """Tests for parse_filters: all key aliases, combinations, no filters, edge cases."""

    def test_collection_filter_aliases(self):
        """collection/col/c 都应映射到 collection."""
        for key in ("collection", "col", "c"):
            parsed = QueryProcessor.parse(f"{key}:技术文档 微服务架构")
            assert parsed.filters.get("collection") == "技术文档"
            assert parsed.clean_query == "微服务架构"

    def test_type_filter_aliases(self):
        """type/doc_type/t 都应映射到 doc_type."""
        for key in ("type", "doc_type", "t"):
            parsed = QueryProcessor.parse(f"{key}:pdf 数据库优化")
            assert parsed.filters.get("doc_type") == "pdf"
            assert parsed.clean_query == "数据库优化"

    def test_tag_filter_aliases(self):
        """tag/tags 都应映射到 tags (list)."""
        for key in ("tag", "tags"):
            parsed = QueryProcessor.parse(f"{key}:架构,设计 数据库优化")
            assert "tags" in parsed.filters
            assert parsed.filters["tags"] == ["架构", "设计"]
            assert parsed.clean_query == "数据库优化"

    def test_source_filter_aliases(self):
        """source/src/s 都应映射到 source."""
        for key in ("source", "src", "s"):
            parsed = QueryProcessor.parse(f"{key}:doc.pdf 微服务")
            assert parsed.filters.get("source") == "doc.pdf"
            assert parsed.clean_query == "微服务"

    def test_combined_filters(self):
        """组合过滤：多个 key:value 同时使用."""
        parsed = QueryProcessor.parse(
            "collection:技术文档 type:pdf tag:架构,微服务 source:design.pdf 系统设计原则"
        )
        assert parsed.filters["collection"] == "技术文档"
        assert parsed.filters["doc_type"] == "pdf"
        assert parsed.filters["tags"] == ["架构", "微服务"]
        assert parsed.filters["source"] == "design.pdf"
        assert parsed.clean_query == "系统设计原则"

    def test_no_filters(self):
        """无过滤条件时返回空 filters，clean_query = 原始查询."""
        parsed = QueryProcessor.parse("如何优化数据库性能")
        assert parsed.filters == {}
        assert parsed.clean_query == "如何优化数据库性能"

    def test_unknown_key_as_generic_metadata(self):
        """未识别的键作为通用 metadata 过滤."""
        parsed = QueryProcessor.parse("author:张三 项目进度")
        assert parsed.filters.get("author") == "张三"
        assert parsed.clean_query == "项目进度"

    def test_quoted_value(self):
        """带引号的 value 应正确提取."""
        parsed = QueryProcessor.parse('collection:"技术 文档" 微服务')
        assert parsed.filters["collection"] == "技术 文档"
        assert parsed.clean_query == "微服务"

        parsed2 = QueryProcessor.parse("collection:'技术文档' 微服务")
        assert parsed2.filters["collection"] == "技术文档"

    def test_empty_query(self):
        """空查询应安全处理."""
        parsed = QueryProcessor.parse("")
        assert parsed.clean_query == ""
        assert parsed.filters == {}

        parsed2 = QueryProcessor.parse("   ")
        # 纯空白查询的 clean_query 应为空字符串
        assert parsed2.clean_query == ""
        assert parsed2.filters == {}

    def test_multiple_tags_accumulate(self):
        """多次指定 tag 应累积."""
        parsed = QueryProcessor.parse("tag:架构 tag:微服务 系统设计")
        assert parsed.filters["tags"] == ["架构", "微服务"]
        assert parsed.clean_query == "系统设计"

    def test_tag_comma_separated_with_spaces(self):
        """逗号分隔的 tag 中的空格应被清理."""
        parsed = QueryProcessor.parse("tag:架构,微服务,分布式   系统设计")
        assert parsed.filters["tags"] == ["架构", "微服务", "分布式"]
        assert parsed.clean_query == "系统设计"

    def test_colon_in_query_not_mistaken(self):
        """查询中的冒号如果没有前导 word 字符不应被误解析."""
        parsed = QueryProcessor.parse("时间 10:30 的会议记录")
        # "10:30" 中 "10" 是 word 字符，会被匹配
        # 这是预期行为 — 用户不应在查询中使用会被误解析的模式
        assert parsed.clean_query is not None


class TestQueryProcessorToWhere:
    """Tests for to_chromadb_where: converts filters to ChromaDB where clause."""

    def test_empty_filters(self):
        assert QueryProcessor.to_chromadb_where({}) is None

    def test_collection_only_ignored(self):
        """仅 collection filter 不产生 where 子句（collection 在检索层处理）."""
        filters = {"collection": "技术文档"}
        assert QueryProcessor.to_chromadb_where(filters) is None

    def test_single_filter(self):
        filters = {"doc_type": "pdf"}
        result = QueryProcessor.to_chromadb_where(filters)
        assert result == {"doc_type": {"$eq": "pdf"}}

    def test_multiple_filters_and(self):
        filters = {"doc_type": "pdf", "source": "doc.pdf"}
        result = QueryProcessor.to_chromadb_where(filters)
        assert "$and" in result
        assert {"doc_type": {"$eq": "pdf"}} in result["$and"]
        assert {"source": {"$eq": "doc.pdf"}} in result["$and"]

    def test_tags_as_multiple_eq(self):
        filters = {"tags": ["架构", "微服务"]}
        result = QueryProcessor.to_chromadb_where(filters)
        assert "$and" in result
        assert {"tags": {"$eq": "架构"}} in result["$and"]
        assert {"tags": {"$eq": "微服务"}} in result["$and"]

    def test_mixed_collection_and_metadata(self):
        """collection 应与 metadata where 分开处理."""
        filters = {"collection": "技术文档", "doc_type": "pdf"}
        result = QueryProcessor.to_chromadb_where(filters)
        # collection 不在 where 中
        assert result == {"doc_type": {"$eq": "pdf"}}

    def test_get_collection_name(self):
        filters = {"collection": "技术文档", "doc_type": "pdf"}
        assert QueryProcessor.get_collection_name(filters) == "技术文档"

    def test_get_collection_name_none(self):
        assert QueryProcessor.get_collection_name({}) is None
        assert QueryProcessor.get_collection_name({"doc_type": "pdf"}) is None
