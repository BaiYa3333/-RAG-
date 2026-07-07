"""Unit tests for ChunkRefiner — Task 8.1."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestChunkRefiner:
    """Tests for ChunkRefiner: normal refinement, LLM failure degradation, empty text."""

    @pytest.fixture
    def sample_chunks(self):
        return [
            {
                "chunk_id": "abc123_p0000_c0000_a1b2c3d4",
                "parent_chunk_id": "abc123_p0000",
                "content": "第 1 页\n\n## 微服务架构概述\n\n微服务是一种将应用拆分为小型、独立服务的架构风格。\n\n版权所有 © 2024",
                "metadata": {"chunk_index": 0, "parent_index": 0},
            },
            {
                "chunk_id": "abc123_p0000_c0001_e5f6g7h8",
                "parent_chunk_id": "abc123_p0000",
                "content": "服务之间通过轻量级通信机制（如 HTTP/REST 或消息队列）进行协作。\n\nPage 2 of 10",
                "metadata": {"chunk_index": 1, "parent_index": 0},
            },
        ]

    def _make_mock_llm(self, return_text: str | None = None, should_fail: bool = False):
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
            chat_response.choices = [
                MagicMock(message=MagicMock(
                    content=return_text or "## 微服务架构概述\n\n微服务是一种将应用拆分为小型、独立服务的架构风格。"
                ))
            ]
            inner.chat.completions.create = AsyncMock(return_value=chat_response)
        client.client = inner
        return client

    @pytest.mark.asyncio
    async def test_normal_refinement(self, sample_chunks):
        """正常精炼：LLM 返回清洗后文本，应替换原始内容."""
        from src.rag.ingestion.refiner import ChunkRefiner

        mock_client = self._make_mock_llm(
            return_text="## 微服务架构概述\n\n微服务是一种将应用拆分为小型、独立服务的架构风格。"
        )

        with patch("src.rag.ingestion.refiner.create_llm", return_value=mock_client):
            refiner = ChunkRefiner(concurrency=2)
            result = await refiner.refine_chunks(sample_chunks)

        assert len(result) == 2
        # 第一个 chunk 应被精炼（噪音已去除）
        assert "第 1 页" not in result[0]["content"]
        assert "版权所有" not in result[0]["content"]
        assert "微服务架构概述" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_llm_failure_degradation(self, sample_chunks):
        """LLM 调用失败时降级为保留原始文本，不阻塞管道."""
        from src.rag.ingestion.refiner import ChunkRefiner

        mock_client = self._make_mock_llm(should_fail=True)

        with patch("src.rag.ingestion.refiner.create_llm", return_value=mock_client):
            refiner = ChunkRefiner(concurrency=2)
            result = await refiner.refine_chunks(sample_chunks)

        assert len(result) == 2
        # 降级后应保留原始内容
        assert "第 1 页" in result[0]["content"]
        assert result[0]["content"] == sample_chunks[0]["content"]

    @pytest.mark.asyncio
    async def test_empty_text_passthrough(self):
        """空文本/空白文本应原样返回."""
        from src.rag.ingestion.refiner import ChunkRefiner

        empty_chunks = [
            {"chunk_id": "x", "content": "", "metadata": {}},
            {"chunk_id": "y", "content": "   ", "metadata": {}},
        ]

        refiner = ChunkRefiner(concurrency=1)
        result = await refiner.refine_chunks(empty_chunks)

        assert len(result) == 2
        assert result[0]["content"] == ""
        assert result[1]["content"] == "   "

    @pytest.mark.asyncio
    async def test_preserves_image_placeholders(self):
        """精炼后应保留 [IMAGE:xxx] 占位符."""
        from src.rag.ingestion.refiner import ChunkRefiner

        mock_client = self._make_mock_llm(
            return_text="架构图如下：\n\n[IMAGE:arch-diagram]"
        )

        chunks = [
            {"chunk_id": "x", "content": "架构图如下：\n\n[IMAGE:arch-diagram]\n\nPage 1", "metadata": {}}
        ]

        with patch("src.rag.ingestion.refiner.create_llm", return_value=mock_client):
            refiner = ChunkRefiner(concurrency=1)
            result = await refiner.refine_chunks(chunks)

        assert "[IMAGE:arch-diagram]" in result[0]["content"]
        assert "Page 1" not in result[0]["content"]

    @pytest.mark.asyncio
    async def test_preserves_markdown_tables(self):
        """精炼后应保留 Markdown 表格结构."""
        from src.rag.ingestion.refiner import ChunkRefiner

        table_content = "| 列A | 列B |\n|-----|-----|\n| 1 | 2 |\n\nPage 3"
        clean_table = "| 列A | 列B |\n|-----|-----|\n| 1 | 2 |"

        mock_client = self._make_mock_llm(return_text=clean_table)

        chunks = [
            {"chunk_id": "x", "content": table_content, "metadata": {}}
        ]

        with patch("src.rag.ingestion.refiner.create_llm", return_value=mock_client):
            refiner = ChunkRefiner(concurrency=1)
            result = await refiner.refine_chunks(chunks)

        assert "| 列A | 列B |" in result[0]["content"]
        assert "Page 3" not in result[0]["content"]
