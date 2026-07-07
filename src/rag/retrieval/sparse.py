"""稀疏检索 — BM25 + jieba 关键词检索（JSON 持久化 + 增量更新）.

BM25 索引持久化为 JSON 文件，支持：
- save_index / load_index: JSON 序列化/反序列化
- 原子写入: .tmp → os.replace
- add_documents: 增量更新（合并新词项，重算 IDF）
- 首次检索自动加载（JSON 优先，回退到 ChromaDB 全量重建）
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone

import jieba
from rank_bm25 import BM25Okapi

from src.config import settings
from src.observability.langfuse_context import langfuse_context

logger = logging.getLogger(__name__)


def _index_dir() -> str:
    return settings.rag_bm25_index_dir or "data/db/bm25/"


def _index_path(collection_name: str) -> str:
    return os.path.join(_index_dir(), f"{collection_name}_bm25.json")


def _tmp_path(collection_name: str) -> str:
    return os.path.join(_index_dir(), f"{collection_name}_bm25.json.tmp")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SparseRetriever:
    """BM25 稀疏检索器 — 支持 JSON 持久化和增量更新."""

    def __init__(self, collection_name: str | None = None):
        self._documents: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._doc_lengths: list[int] = []
        self._document_frequency: dict[str, int] = {}
        self._avg_doc_length: float = 0.0
        self._collection_name = collection_name
        # Track whether we've loaded/persisted this index
        self._loaded_from_disk: bool = False
        self._file_mtime: float | None = None

    # ── Public API ───────────────────────────────────────────

    def build_index(self, documents: list[dict]) -> None:
        """从文档列表构建 BM25 索引（全量）。"""
        self._documents = documents
        tokenized = [_tokenize(d["content"]) for d in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._doc_lengths = [len(t) for t in tokenized]
        self._document_frequency = _compute_df(tokenized)
        self._avg_doc_length = (sum(self._doc_lengths) / len(self._doc_lengths)
                                if self._doc_lengths else 0.0)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25 检索.

        如果是首次检索且有 collection_name，尝试从 JSON 文件加载索引。
        """
        if not self._bm25:
            # 尝试从磁盘加载（懒加载）
            if self._collection_name and not self._loaded_from_disk:
                if self._try_load_from_disk():
                    pass  # 加载成功，_bm25 已重建
                else:
                    return []
            else:
                return []

        # 防御性检查：load 之后 _bm25 必须非空
        if not self._bm25:
            logger.warning("bm25_search_failed: _bm25 is None after load attempt")
            return []

        start_time = time.monotonic()
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = [
            {
                "id": self._documents[i].get("id", str(i)),
                "content": self._documents[i]["content"],
                "metadata": self._documents[i].get("metadata", {}),
                "score": float(score),
                "source": "sparse",
            }
            for i, score in ranked if score > 0
        ]

        latency_ms = (time.monotonic() - start_time) * 1000
        try:
            langfuse_context.update_current_observation(
                retrieval_type="bm25",
                top_k=top_k,
                recall_count=len(results),
                latency_ms=round(latency_ms, 2),
            )
        except Exception:
            pass

        return results

    # ── JSON 持久化 ─────────────────────────────────────────

    def save_index(self, collection_name: str | None = None) -> bool:
        """将 BM25 索引持久化为 JSON 文件（原子写入）.

        Args:
            collection_name: ChromaDB collection 名。若未提供，使用实例的 _collection_name。

        Returns:
            True if saved successfully, False otherwise.
        """
        col = collection_name or self._collection_name
        if not col:
            logger.warning("bm25_save_skipped: no collection_name")
            return False

        t0 = time.monotonic()
        os.makedirs(_index_dir(), exist_ok=True)

        data = {
            "collection": col,
            "num_docs": len(self._documents),
            "avg_doc_length": self._avg_doc_length,
            "document_frequency": self._document_frequency,
            "doc_lengths": self._doc_lengths,
            "documents": self._documents,
            "updated_at": _now_iso(),
        }

        tmp = _tmp_path(col)
        target = _index_path(col)

        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, target)  # 原子重命名
            elapsed_ms = (time.monotonic() - t0) * 1000
            file_size = os.path.getsize(target)
            logger.info(
                "bm25_saved",
                collection=col,
                num_docs=len(self._documents),
                file_size=file_size,
                elapsed_ms=round(elapsed_ms, 2),
            )
            self._file_mtime = os.path.getmtime(target)
            return True
        except Exception as exc:
            logger.warning("bm25_save_failed", collection=col, error=str(exc))
            # 清理临时文件
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            return False

    def load_index(self, collection_name: str | None = None) -> bool:
        """从 JSON 文件加载 BM25 索引.

        Args:
            collection_name: ChromaDB collection 名。若未提供，使用实例的 _collection_name。

        Returns:
            True if loaded successfully, False if file not found or corrupt.
        """
        col = collection_name or self._collection_name
        if not col:
            return False

        t0 = time.monotonic()
        path = _index_path(col)

        if not os.path.exists(path):
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 恢复文档（v2 持久化中包含 documents 字段）
            documents = data.get("documents", [])
            if not documents:
                logger.warning("bm25_load_empty", collection=col)
                return False

            self._documents = documents
            self._document_frequency = data.get("document_frequency", {})
            self._doc_lengths = data.get("doc_lengths", [])
            self._avg_doc_length = data.get("avg_doc_length", 0.0)

            # 从恢复的文档重建 BM25Okapi 对象
            tokenized = [_tokenize(d["content"]) for d in self._documents]
            self._bm25 = BM25Okapi(tokenized)

            self._collection_name = col
            self._loaded_from_disk = True

            elapsed_ms = (time.monotonic() - t0) * 1000
            file_size = os.path.getsize(path)
            self._file_mtime = os.path.getmtime(path)
            logger.info(
                "bm25_loaded",
                collection=col,
                num_docs=len(self._documents),
                file_size=file_size,
                elapsed_ms=round(elapsed_ms, 2),
            )
            return True
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("bm25_load_corrupted", collection=col, error=str(exc))
            return False

    def _try_load_from_disk(self) -> bool:
        """尝试从磁盘加载完整索引（含文档和 BM25Okapi 对象）。

        如果 JSON 文件不存在或损坏，返回 False 让调用方执行全量重建。
        加载成功后 _loaded_from_disk 设为 True，_bm25 已重建可用。
        """
        try:
            if self.load_index(self._collection_name):
                # load_index 内部已设置 _loaded_from_disk 和 _bm25
                return True
        except Exception as exc:
            logger.warning("bm25_try_load_error: %s", exc)
        # 防止无限重试 — 即使加载失败也标记已尝试
        self._loaded_from_disk = True
        return False

    # ── 增量更新 ────────────────────────────────────────────

    def add_documents(self, new_docs: list[dict]) -> None:
        """增量添加新文档到 BM25 索引.

        合并新词项统计，更新 IDF 和文档长度，重写索引文件。
        这比从 ChromaDB 全量重建要高效得多。

        Args:
            new_docs: 新增文档列表 [{"content": ..., "id": ..., "metadata": {...}}, ...]
        """
        if not new_docs:
            return

        t0 = time.monotonic()
        old_count = len(self._documents)
        new_tokenized = [_tokenize(d["content"]) for d in new_docs]

        # 合并文档
        self._documents.extend(new_docs)

        # 更新文档长度
        new_lengths = [len(t) for t in new_tokenized]
        self._doc_lengths.extend(new_lengths)

        # 更新平均长度
        total_docs = len(self._doc_lengths)
        total_length = sum(self._doc_lengths)
        self._avg_doc_length = total_length / total_docs if total_docs > 0 else 0.0

        # 合并词频统计
        new_df = _compute_df(new_tokenized)
        for term, count in new_df.items():
            self._document_frequency[term] = self._document_frequency.get(term, 0) + count

        # 重建 BM25Okapi 对象（rank_bm25 需要全量 tokenized 文档）
        all_tokenized = [_tokenize(d["content"]) for d in self._documents]
        self._bm25 = BM25Okapi(all_tokenized)

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "bm25_incremental_update",
            collection=self._collection_name,
            old_count=old_count,
            new_count=len(new_docs),
            total=len(self._documents),
            elapsed_ms=round(elapsed_ms, 2),
        )

        # 更新后自动持久化
        if self._collection_name:
            self.save_index(self._collection_name)


# ── 模块级辅助函数 ───────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """jieba 中文分词."""
    return [w.strip() for w in jieba.cut(text) if w.strip()]


def _compute_df(tokenized_docs: list[list[str]]) -> dict[str, int]:
    """计算文档频率（document frequency）：每个 term 出现在多少个文档中."""
    df: dict[str, int] = {}
    for tokens in tokenized_docs:
        unique_terms = set(tokens)
        for term in unique_terms:
            df[term] = df.get(term, 0) + 1
    return df
