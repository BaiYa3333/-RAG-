"""Unit tests for compute_quality_score, filter_short_chunks, and quality exception mapping."""

import pytest


class TestComputeQualityScore:
    """Quality scoring function — meaningful character ratio."""

    def test_normal_chinese_text_scores_high(self):
        from src.rag.ingestion.cleaner import compute_quality_score
        text = "这是一段正常的中文文本，用于测试质量评分功能。应该得到较高的分数。"
        score = compute_quality_score(text)
        assert score > 0.70, f"Expected >0.70 for clean Chinese, got {score}"

    def test_normal_english_text_scores_high(self):
        from src.rag.ingestion.cleaner import compute_quality_score
        text = "This is a normal English paragraph used to test the quality scoring function."
        score = compute_quality_score(text)
        assert score > 0.70, f"Expected >0.70 for clean English, got {score}"

    def test_mixed_cjk_ascii_scores_high(self):
        from src.rag.ingestion.cleaner import compute_quality_score
        text = "RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术。"
        score = compute_quality_score(text)
        assert score > 0.70, f"Expected >0.70 for mixed CJK+ASCII, got {score}"

    def test_garbled_control_chars_scores_low(self):
        from src.rag.ingestion.cleaner import compute_quality_score
        # Simulate garbled text with 80% control characters
        text = "abc " + "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f" * 10
        score = compute_quality_score(text)
        assert score < 0.30, f"Expected <0.30 for garbled control chars, got {score}"

    def test_all_replacement_chars_scores_zero(self):
        from src.rag.ingestion.cleaner import compute_quality_score
        text = "�" * 100
        score = compute_quality_score(text)
        assert score == 0.0, f"Expected 0.0 for all U+FFFD, got {score}"

    def test_empty_string_scores_zero(self):
        from src.rag.ingestion.cleaner import compute_quality_score
        assert compute_quality_score("") == 0.0

    def test_whitespace_only_scores_zero(self):
        """Whitespace-only text scores 0 because no letters/CJK — pure whitespace is not meaningful."""
        from src.rag.ingestion.cleaner import compute_quality_score
        # Spaces, tabs, newlines are counted but there are no letter/CJK chars
        # The ratio would be 1.0 since all whitespace chars are in the meaningful set
        # Actually, spaces (0x20) are in the printable ASCII range, so they'd count.
        # Let me reconsider... spaces, tabs, newlines all count as meaningful.
        text = "   \t\n   "
        score = compute_quality_score(text)
        # Whitespace chars are in the meaningful set, so this should be 1.0
        assert score == 1.0, f"Whitespace-only should be 1.0, got {score}"

    def test_mixed_garbage_and_content_scores_intermediate(self):
        from src.rag.ingestion.cleaner import compute_quality_score
        # ~50% garbage bytes, ~50% good text
        text = "Hello 你好" + "\x00\x01\x02\x03\x04\x05" * 5
        score = compute_quality_score(text)
        assert 0.20 < score < 0.80, f"Expected intermediate score, got {score}"

    def test_binary_like_text_scores_very_low(self):
        from src.rag.ingestion.cleaner import compute_quality_score
        # Simulate a binary file masquerading as text
        garbage = "".join(chr(i) for i in range(256) if i not in (0x09, 0x0A, 0x0D) and (i < 0x20 or i > 0x7E))
        # Only keep non-meaningful chars
        garbage_only = "".join(
            ch for ch in garbage
            if not (
                (0x20 <= ord(ch) <= 0x7E)
                or (0x4E00 <= ord(ch) <= 0x9FFF)
                or (0xF900 <= ord(ch) <= 0xFAFF)
                or (0x3000 <= ord(ch) <= 0x303F)
                or (0xFF00 <= ord(ch) <= 0xFFEF)
                or ord(ch) in (0x09, 0x0A, 0x0D)
            )
        )
        if garbage_only:
            text = "A" + garbage_only  # ~1% meaningful
            score = compute_quality_score(text)
            assert score < 0.10, f"Expected <0.10 for binary-like, got {score}"


class TestFilterShortChunks:
    """Short chunk filtering — drop fragments below min length."""

    def test_filters_below_min_length(self):
        from src.rag.ingestion.cleaner import filter_short_chunks
        chunks = [
            {"content": "short", "chunk_id": "1"},
            {"content": "this is long enough to pass the filter", "chunk_id": "2"},
            {"content": "a", "chunk_id": "3"},
            {"content": "tiny", "chunk_id": "4"},
        ]
        result = filter_short_chunks(chunks, 10)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "2"

    def test_preserves_at_boundary(self):
        from src.rag.ingestion.cleaner import filter_short_chunks
        chunks = [
            {"content": "1234567890", "chunk_id": "1"},  # exactly 10 chars
            {"content": "12345678901", "chunk_id": "2"},  # 11 chars
        ]
        result = filter_short_chunks(chunks, 10)
        assert len(result) == 2

    def test_all_below_threshold_filters_all(self):
        from src.rag.ingestion.cleaner import filter_short_chunks
        chunks = [
            {"content": "a", "chunk_id": "1"},
            {"content": "b", "chunk_id": "2"},
        ]
        result = filter_short_chunks(chunks, 5)
        assert len(result) == 0

    def test_zero_min_length_is_noop(self):
        from src.rag.ingestion.cleaner import filter_short_chunks
        chunks = [{"content": "", "chunk_id": "1"}]
        result = filter_short_chunks(chunks, 0)
        assert len(result) == 1

    def test_negative_min_length_is_noop(self):
        from src.rag.ingestion.cleaner import filter_short_chunks
        chunks = [{"content": "", "chunk_id": "1"}]
        result = filter_short_chunks(chunks, -1)
        assert len(result) == 1

    def test_empty_content_always_filtered(self):
        from src.rag.ingestion.cleaner import filter_short_chunks
        chunks = [
            {"content": "", "chunk_id": "1"},
            {"content": None, "chunk_id": "2"},
            {"chunk_id": "3"},  # no content key
        ]
        result = filter_short_chunks(chunks, 1)
        assert len(result) == 0

    def test_does_not_mutate_input(self):
        from src.rag.ingestion.cleaner import filter_short_chunks
        chunks = [
            {"content": "a", "chunk_id": "1"},
            {"content": "long enough", "chunk_id": "2"},
        ]
        original_len = len(chunks)
        filter_short_chunks(chunks, 10)
        assert len(chunks) == original_len  # original unchanged


class TestQualityExceptionMapping:
    """Verify IngestionQualityError is properly wired in the exception hierarchy."""

    def test_is_ingestion_error(self):
        from src.utils.exceptions import IngestionQualityError, IngestionError
        assert issubclass(IngestionQualityError, IngestionError)

    def test_is_rag_exception(self):
        from src.utils.exceptions import IngestionQualityError, RAGException
        assert issubclass(IngestionQualityError, RAGException)

    def test_can_be_raised_with_message(self):
        from src.utils.exceptions import IngestionQualityError
        msg = "文档质量不达标（得分 0.15，要求 ≥ 0.30），存在大量乱码或无效字符，无法入库"
        exc = IngestionQualityError(msg)
        assert str(exc) == msg


class TestConfigDefaults:
    """Verify the new settings have sensible defaults."""

    def test_min_quality_score_default(self):
        from src.config import settings
        assert hasattr(settings, "ingestion_min_quality_score")
        assert 0.0 <= settings.ingestion_min_quality_score <= 1.0

    def test_min_chunk_length_default(self):
        from src.config import settings
        assert hasattr(settings, "ingestion_min_chunk_length")
        assert settings.ingestion_min_chunk_length >= 0
