"""Unit tests for estimate_tokens in context_manager.py (Task 5.2)."""

import pytest


class TestEstimateTokens:
    """Task 5.2: Chinese-aware token estimation tests."""

    def test_pure_chinese(self):
        """Pure Chinese: each CJK char = 1 token."""
        from src.memory.context_manager import estimate_tokens

        # 17 CJK chars: "量子纠缠是物理学中最神秘的现象之一"
        text = "量子纠缠是物理学中最神秘的现象之一"
        tokens = estimate_tokens(text)
        assert tokens == 17

    def test_pure_english(self):
        """Pure English: 4 chars ≈ 1 token."""
        from src.memory.context_manager import estimate_tokens

        # "The quick brown fox jumps over the lazy dog"
        # 43 letters + spaces = 43 chars total
        # 43 // 4 = 10, but with max(1, ...) = 10
        text = "The quick brown fox jumps over the lazy dog"
        tokens = estimate_tokens(text)
        # 43 chars // 4 = 10, max(1, 10) = 10
        assert tokens == 10

    def test_mixed_chinese_english(self):
        """Mixed text: CJK chars counted separately from Latin chars."""
        from src.memory.context_manager import estimate_tokens

        # "RAG (检索增强生成) is a technique"
        # CJK: 检索增强生成 = 7 chars = 7 tokens
        # Other: "RAG ( ) is a technique" = 23 chars
        # 23 // 4 = 5, max(1, 5) = 5
        # Total: 7 + 5 = 12
        text = "RAG (检索增强生成) is a technique for improving LLM outputs"
        tokens = estimate_tokens(text)
        # CJK: 检索增强生成 = 7 chars → 7 tokens
        # Other: "RAG ( ) is a technique for improving LLM outputs" = 53 chars
        # 53 // 4 = 13
        # Total: 7 + 13 = 20
        assert tokens > 0
        # Verify CJK counting works
        cjk_count = sum(1 for ch in text if '一' <= ch <= '鿿')
        other_count = len(text) - cjk_count
        expected = cjk_count + max(1, other_count // 4)
        assert tokens == expected

    def test_empty_input(self):
        """Empty input should return 0."""
        from src.memory.context_manager import estimate_tokens

        assert estimate_tokens("") == 0

    def test_short_english_returns_one(self):
        """Short English text should return at least 1 token."""
        from src.memory.context_manager import estimate_tokens

        assert estimate_tokens("Hi") == 1
        assert estimate_tokens("abc") == 1

    def test_punctuation_counted(self):
        """Punctuation and spaces are counted as non-CJK chars."""
        from src.memory.context_manager import estimate_tokens

        # "你好！" = 2 CJK + 1 other = 2 + max(1, 1//4) = 2 + 1 = 3
        text = "你好！"
        tokens = estimate_tokens(text)
        assert tokens == 3
