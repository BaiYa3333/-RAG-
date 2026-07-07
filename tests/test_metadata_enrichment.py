"""Unit tests for MetadataEnricher — Task 8.2."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMetadataEnricher:
    """Tests for MetadataEnricher: normal generation, LLM failure, JSON parsing errors."""

    @pytest.fixture
    def sample_chunks(self):
        return [
            {
                "chunk_id": "abc123_p0000_c0000_a1b2c3d4",
                "parent_chunk_id": "abc123_p0000",
                "content": "微服务架构是一种将应用拆分为小型独立服务的架构风格。每个服务围绕业务能力构建，可独立部署和扩展。",
                "metadata": {"chunk_index": 0, "source": "doc.pdf"},
            },
        ]

    def _make_mock_llm(self, return_content: str | None = None, should_fail: bool = False):
        """Create a mock LLM client."""
        client = MagicMock()
        client.model_id = "deepseek-chat"
        client.default_params = {"temperature": 0.7, "max_tokens": 4096, "top_p": 0.9}

        inner = MagicMock()
        if should_fail:
            inner.chat.completions.create = AsyncMock(
                side_effect=Exception("API error")
            )
        else:
            chat_response = MagicMock()
            if return_content is None:
                return_content = json.dumps({
                    "title": "微服务架构概述",
                    "summary": "微服务是一种将应用拆分为小型、独立服务的架构风格，每个服务围绕业务能力构建，可独立部署和扩展。",
                    "tags": ["微服务", "架构", "分布式系统", "服务拆分", "部署"],
                }, ensure_ascii=False)
            chat_response.choices = [
                MagicMock(message=MagicMock(content=return_content))
            ]
            inner.chat.completions.create = AsyncMock(return_value=chat_response)
        client.client = inner
        return client

    @pytest.mark.asyncio
    async def test_normal_enrichment(self, sample_chunks):
        """正常生成：LLM 返回 Title/Summary/Tags，应正确拼接到 content."""
        from src.rag.ingestion.enricher import MetadataEnricher

        mock_client = self._make_mock_llm()

        with patch("src.rag.ingestion.enricher.create_llm", return_value=mock_client):
            enricher = MetadataEnricher(concurrency=1)
            result = await enricher.enrich_chunks(sample_chunks)

        assert len(result) == 1
        chunk = result[0]

        assert chunk["content"].startswith("# 微服务架构概述")
        assert "微服务是一种将应用拆分为小型、独立服务的架构风格" in chunk["content"]
        assert "每个服务围绕业务能力构建" in chunk["content"]

        assert "tags" in chunk["metadata"]
        assert len(chunk["metadata"]["tags"]) >= 3
        assert "微服务" in chunk["metadata"]["tags"]
        assert chunk["metadata"]["enrichment_title"] == "微服务架构概述"

    @pytest.mark.asyncio
    async def test_llm_failure_degradation(self, sample_chunks):
        """LLM 调用失败时降级为保留原始 chunk."""
        from src.rag.ingestion.enricher import MetadataEnricher

        mock_client = self._make_mock_llm(should_fail=True)

        with patch("src.rag.ingestion.enricher.create_llm", return_value=mock_client):
            enricher = MetadataEnricher(concurrency=1)
            result = await enricher.enrich_chunks(sample_chunks)

        assert len(result) == 1
        assert result[0]["content"] == sample_chunks[0]["content"]

    @pytest.mark.asyncio
    async def test_json_parse_error_degradation(self, sample_chunks):
        """LLM 返回无效 JSON 时降级为保留原始 chunk."""
        from src.rag.ingestion.enricher import MetadataEnricher

        mock_client = self._make_mock_llm(return_content="这是一段无效的输出，不是 JSON 格式")

        with patch("src.rag.ingestion.enricher.create_llm", return_value=mock_client):
            enricher = MetadataEnricher(concurrency=1)
            result = await enricher.enrich_chunks(sample_chunks)

        assert len(result) == 1
        assert result[0]["content"] == sample_chunks[0]["content"]

    @pytest.mark.asyncio
    async def test_json_in_code_block(self, sample_chunks):
        """JSON 包裹在 ```json ... ``` 代码块中应能正确解析."""
        from src.rag.ingestion.enricher import MetadataEnricher

        metadata_json = json.dumps({
            "title": "测试标题",
            "summary": "测试摘要内容。",
            "tags": ["标签1", "标签2", "标签3"],
        }, ensure_ascii=False)
        mock_client = self._make_mock_llm(return_content=f"```json\n{metadata_json}\n```")

        with patch("src.rag.ingestion.enricher.create_llm", return_value=mock_client):
            enricher = MetadataEnricher(concurrency=1)
            result = await enricher.enrich_chunks(sample_chunks)

        assert len(result) == 1
        assert result[0]["metadata"]["enrichment_title"] == "测试标题"
        assert result[0]["metadata"]["tags"] == ["标签1", "标签2", "标签3"]

    @pytest.mark.asyncio
    async def test_tags_limit_10(self, sample_chunks):
        """Tags 超过 10 个时应截断."""
        from src.rag.ingestion.enricher import MetadataEnricher

        metadata_json = json.dumps({
            "title": "测试",
            "summary": "测试摘要。",
            "tags": ["tag" + str(i) for i in range(15)],
        }, ensure_ascii=False)
        mock_client = self._make_mock_llm(return_content=metadata_json)

        with patch("src.rag.ingestion.enricher.create_llm", return_value=mock_client):
            enricher = MetadataEnricher(concurrency=1)
            result = await enricher.enrich_chunks(sample_chunks)

        assert len(result) == 1
        assert len(result[0]["metadata"]["tags"]) <= 10

    @pytest.mark.asyncio
    async def test_empty_chunks_passthrough(self):
        """空 chunk 列表应原样返回."""
        from src.rag.ingestion.enricher import MetadataEnricher

        enricher = MetadataEnricher(concurrency=1)
        result = await enricher.enrich_chunks([])

        assert result == []
