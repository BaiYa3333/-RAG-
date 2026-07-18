"""稀疏检索 — BM25 + jieba 关键词检索（倒排索引 + JSON 持久化 + 增量更新）.

BM25 使用自建倒排索引（inverted index）实现，JSON 文件中存储：
- inverted_index: term → {doc_idx: tf}  倒排索引 / 词频
- idf: term → idf_value                 逆文档频率
- doc_lengths: [int, ...]               每个文档的长度
- avg_doc_length: float                 平均文档长度

不再依赖 rank_bm25 库，BM25 评分手工计算。

特性：
- save_index / load_index: JSON 序列化/反序列化
- 原子写入: .tmp → os.replace
- add_documents: 增量更新（合并新词项，更新倒排索引 + 重算 IDF）
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

from src.config import settings
from src.observability.langfuse_context import langfuse_context

logger = logging.getLogger(__name__)

# ── BM25 参数 ────────────────────────────────────────────────
BM25_K1 = 1.5   # 词频饱和度参数
BM25_B = 0.75   # 文档长度归一化参数


def _index_dir() -> str:
    return settings.rag_bm25_index_dir or "data/db/bm25/"


def _index_path(collection_name: str) -> str:
    return os.path.join(_index_dir(), f"{collection_name}_bm25.json")


def _tmp_path(collection_name: str) -> str:
    return os.path.join(_index_dir(), f"{collection_name}_bm25.json.tmp")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SparseRetriever:
    """BM25 稀疏检索器 — 基于自建倒排索引，支持 JSON 持久化和增量更新."""

    def __init__(self, collection_name: str | None = None):
        self._documents: list[dict] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0

        # 倒排索引: term → {doc_idx: term_frequency}
        self._inverted_index: dict[str, dict[int, int]] = {}

        # 文档频率: term → df (包含该词项的文档数)
        self._document_frequency: dict[str, int] = {}

        # 预计算的 IDF: term → idf_value
        self._idf: dict[str, float] = {}

        self._collection_name = collection_name
        self._loaded_from_disk: bool = False
        self._file_mtime: float | None = None

    # ── Public API ───────────────────────────────────────────

    def build_index(self, documents: list[dict]) -> None:
        """从文档列表构建倒排索引和 IDF（全量）。"""
        self._documents = documents
        tokenized = [_tokenize(d["content"]) for d in documents]
        self._doc_lengths = [len(t) for t in tokenized]
        total_len = sum(self._doc_lengths)
        num_docs = len(self._doc_lengths)
        self._avg_doc_length = total_len / num_docs if num_docs > 0 else 0.0

        # 构建倒排索引和文档频率
        self._inverted_index = {}
        self._document_frequency = {}

        for doc_idx, tokens in enumerate(tokenized):
            tf = dict(Counter(tokens))
            for term, freq in tf.items():
                # 倒排索引
                if term not in self._inverted_index:
                    self._inverted_index[term] = {}
                self._inverted_index[term][doc_idx] = freq
                # 文档频率
                self._document_frequency[term] = self._document_frequency.get(term, 0) + 1

        # 预计算 IDF
        self._compute_idf()

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25 检索。

        如果是首次检索且有 collection_name，尝试从 JSON 文件加载索引。
        """
        if not self._inverted_index and not self._document_frequency:
            # 尝试从磁盘加载（懒加载）
            if self._collection_name and not self._loaded_from_disk:
                if self._try_load_from_disk():
                    pass  # 加载成功
                else:
                    return []
            else:
                return []

        # 防御性检查
        if not self._inverted_index and not self._document_frequency:
            logger.warning("bm25_search_failed: no index data available")
            return []

        start_time = time.monotonic()
        tokens = _tokenize(query)

        # 手工计算每个文档的 BM25 分数
        num_docs = len(self._documents)
        if num_docs == 0:
            return []

        scores = [0.0] * num_docs
        for token in set(tokens):  # 查询中每个唯一 term
            if token not in self._inverted_index:
                continue
            idf = self._idf.get(token, 0.0)
            if idf == 0.0:
                continue
            postings = self._inverted_index[token]
            for doc_idx, tf in postings.items():
                doc_len = self._doc_lengths[doc_idx] if doc_idx < len(self._doc_lengths) else 0
                # BM25 评分公式
                numerator = tf * (BM25_K1 + 1)
                denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / max(self._avg_doc_length, 1))
                scores[doc_idx] += idf * numerator / denominator

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
        """将 BM25 索引（倒排索引 + IDF + 文档元数据）持久化为 JSON 文件（原子写入）。

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

        # 序列化 inverted_index：key 为 term，value 为 {doc_idx: tf}
        # 将 int key 转为 str 以兼容 JSON
        inverted_serialized = {
            term: {str(doc_idx): tf for doc_idx, tf in postings.items()}
            for term, postings in self._inverted_index.items()
        }

        data = {
            "collection": col,
            "num_docs": len(self._documents),
            "avg_doc_length": self._avg_doc_length,
            "doc_lengths": self._doc_lengths,
            "document_frequency": self._document_frequency,
            "inverted_index": inverted_serialized,
            "idf": self._idf,
            "documents": self._documents,
            "updated_at": _now_iso(),
        }

        tmp = _tmp_path(col)
        target = _index_path(col)

        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, target)
            elapsed_ms = (time.monotonic() - t0) * 1000
            file_size = os.path.getsize(target)
            logger.info(
                "bm25_saved: collection=%s num_docs=%d num_terms=%d file_size=%d elapsed_ms=%.2f",
                col, len(self._documents), len(self._inverted_index), file_size, elapsed_ms,
            )
            self._file_mtime = os.path.getmtime(target)
            return True
        except Exception as exc:
            logger.warning("bm25_save_failed: collection=%s error=%s", col, exc)
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            return False

    def load_index(self, collection_name: str | None = None) -> bool:
        """从 JSON 文件加载 BM25 索引（倒排索引 + IDF + 文档）。

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

            documents = data.get("documents", [])
            if not documents:
                logger.warning("bm25_load_empty: collection=%s", col)
                return False

            self._documents = documents
            self._document_frequency = data.get("document_frequency", {})
            self._doc_lengths = data.get("doc_lengths", [])
            self._avg_doc_length = data.get("avg_doc_length", 0.0)

            # 恢复倒排索引（str key → int key）
            inverted_raw = data.get("inverted_index", {})
            if inverted_raw:
                self._inverted_index = {
                    term: {int(doc_idx): tf for doc_idx, tf in postings.items()}
                    for term, postings in inverted_raw.items()
                }
            else:
                # 兼容旧格式：从 document_frequency 重建倒排索引
                self._inverted_index = {}
                tokenized = [_tokenize(d["content"]) for d in self._documents]
                for doc_idx, tokens in enumerate(tokenized):
                    tf = dict(Counter(tokens))
                    for term, freq in tf.items():
                        if term not in self._inverted_index:
                            self._inverted_index[term] = {}
                        self._inverted_index[term][doc_idx] = freq

            # 恢复或重算 IDF
            idf_raw = data.get("idf", {})
            if idf_raw:
                self._idf = idf_raw
            else:
                # 兼容旧格式
                self._compute_idf()

            self._collection_name = col
            self._loaded_from_disk = True

            elapsed_ms = (time.monotonic() - t0) * 1000
            file_size = os.path.getsize(path)
            self._file_mtime = os.path.getmtime(path)
            logger.info(
                "bm25_loaded: collection=%s num_docs=%d num_terms=%d file_size=%d elapsed_ms=%.2f",
                col, len(self._documents), len(self._inverted_index), file_size, elapsed_ms,
            )
            return True
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("bm25_load_corrupted: collection=%s error=%s", col, exc)
            return False

    def _try_load_from_disk(self) -> bool:
        """尝试从磁盘加载完整索引（含倒排索引、IDF 和文档）。

        如果 JSON 文件不存在或损坏，返回 False 让调用方执行全量重建。
        """
        try:
            if self.load_index(self._collection_name):
                return True
        except Exception as exc:
            logger.warning("bm25_try_load_error: %s", exc)
        self._loaded_from_disk = True
        return False

    # ── 增量更新 ────────────────────────────────────────────

    def add_documents(self, new_docs: list[dict]) -> None:
        """增量添加新文档到倒排索引。

        合并新词项统计，更新倒排索引 + 文档频率 + IDF + 文档长度，
        重写索引文件。

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

        # 增量更新倒排索引和文档频率
        for doc_idx_offset, tokens in enumerate(new_tokenized):
            doc_idx = old_count + doc_idx_offset
            tf = dict(Counter(tokens))
            for term, freq in tf.items():
                # 倒排索引
                if term not in self._inverted_index:
                    self._inverted_index[term] = {}
                self._inverted_index[term][doc_idx] = freq
                # 文档频率
                self._document_frequency[term] = self._document_frequency.get(term, 0) + 1

        # 重算 IDF（因为 N 和 df 都变了）
        self._compute_idf()

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "bm25_incremental_update: collection=%s old_count=%d new_count=%d total=%d elapsed_ms=%.2f",
            self._collection_name, old_count, len(new_docs), len(self._documents), elapsed_ms,
        )

        # 更新后自动持久化
        if self._collection_name:
            self.save_index(self._collection_name)

    # ── 内部辅助 ─────────────────────────────────────────────

    def _compute_idf(self) -> None:
        """预计算所有词项的 IDF 值（Robertson-Sparck Jones 公式）。"""
        num_docs = len(self._documents)
        if num_docs == 0:
            self._idf = {}
            return

        self._idf = {}
        for term, df in self._document_frequency.items():
            # RSJ IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
            self._idf[term] = idf


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
