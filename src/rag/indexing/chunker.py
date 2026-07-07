"""父子分块器 — 子块 256字符 索引，父块 1024字符 上下文；支持确定性 Chunk ID 和表格语义保留."""

import hashlib
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.rag.ingestion.table_parser import TableParser


def _compute_content_hash(text: str) -> str:
    """计算文本内容的 SHA-256 前 8 位（十六进制）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _generate_deterministic_id(
    doc_hash: str,
    parent_idx: int,
    child_idx: int,
    content_hash: str,
) -> str:
    """生成确定性 Chunk ID: {doc_hash[:12]}_{parent_idx:04d}_{child_idx:04d}_{content_hash[:8]}"""
    return f"{doc_hash[:12]}_p{parent_idx:04d}_c{child_idx:04d}_{content_hash}"


def _generate_deterministic_parent_id(doc_hash: str, parent_idx: int) -> str:
    """生成确定性 Parent ID: {doc_hash[:12]}_p{parent_idx:04d}"""
    return f"{doc_hash[:12]}_p{parent_idx:04d}"


class ParentChildChunker:
    def __init__(self, child_size: int | None = None, parent_size: int | None = None,
                 overlap: int | None = None, splitter_strategy: str | None = None):
        self.child_size = child_size or settings.chunk_size or 256
        self.parent_size = parent_size or settings.parent_chunk_size  # 1024
        # overlap 默认为 child_size 的 ~20%
        _default_overlap = max(1, int(self.child_size * 0.20))
        self.overlap = overlap or settings.chunk_overlap or _default_overlap
        self.splitter_strategy = splitter_strategy or settings.rag_chunk_splitter or "recursive"

        # 优先在 markdown/中文标题处切割，保证标题和其下内容在同一 child 块内
        # 分隔符优先级: Markdown headings → paragraphs → sentences (中/英) → char fallback
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_size,
            chunk_overlap=self.overlap,
            separators=[
                "\n## ", "\n# ",          # markdown 标题
                "\n**",                    # 加粗标题（中文文档常见格式）
                "\n\n", "\n",             # 段落/行边界
                "。", ". ",               # 句子边界（中文句号 / 英文句号+空格）
                " ", ""                   # 词语级 / 字符级回退
            ],
        )
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_size,
            chunk_overlap=self.overlap,
            separators=[
                "\n## ", "\n# ",
                "\n**",
                "\n\n", "\n",
                "。", ". ",
                " ", ""
            ],
        )

    def split_documents(
        self, docs: list[dict], doc_hash: str | None = None,
    ) -> list[dict]:
        """拆分文档为父子 chunk。

        Args:
            docs: 文档元素列表 [{"content": ..., "metadata": {...}}, ...]
            doc_hash: 可选的文档 SHA-256（用于确定性 ID）。若未提供，从 metadata 推导。

        Returns:
            chunk 列表，每个包含 chunk_id, parent_chunk_id, content, metadata。
        """
        all_chunks = []
        use_deterministic = settings.ingestion_deterministic_ids

        for doc in docs:
            content = doc["content"]
            metadata = doc.get("metadata", {})

            # 推导 doc_hash（优先参数，其次 metadata，最后用 content hash fallback）
            _doc_hash = doc_hash or metadata.get("file_hash", "")
            if not _doc_hash and use_deterministic:
                _doc_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            # ── 表格检测（原子边界标记） ──
            table_spans = TableParser.detect_tables(content) if settings.ingestion_table_semantic else []

            parent_chunks = self._parent_splitter.split_text(content)

            for pi, parent_text in enumerate(parent_chunks):
                child_texts = self._child_splitter.split_text(parent_text)

                for ci, child_text in enumerate(child_texts):
                    child_content_hash = _compute_content_hash(child_text)

                    if use_deterministic and _doc_hash:
                        chunk_id = _generate_deterministic_id(
                            _doc_hash, pi, ci, child_content_hash,
                        )
                        parent_id = _generate_deterministic_parent_id(_doc_hash, pi)
                    else:
                        chunk_id = str(uuid.uuid4())
                        parent_id = str(uuid.uuid4())

                    # ── 表格语义保留 ──
                    chunk_content = child_text
                    chunk_meta = {
                        **metadata,
                        "chunk_index": ci,
                        "parent_index": pi,
                        "child_char_count": len(child_text),
                        "parent_char_count": len(parent_text),
                        "content_hash": child_content_hash,
                        "doc_hash": _doc_hash,
                        # 存储父块全文，检索时用于扩展上下文
                        "parent_content": parent_text,
                    }

                    if settings.ingestion_table_semantic and table_spans:
                        # 检查该 child chunk 是否包含表格内容
                        for _start, _end, table_text in table_spans:
                            if table_text in child_text:
                                # 追加 NL 描述
                                desc = TableParser.generate_table_description(table_text)
                                if desc:
                                    chunk_content = child_text + "\n\n" + desc
                                # 合并表格元数据
                                table_meta = TableParser.extract_table_metadata(table_text)
                                chunk_meta.update(table_meta)
                                break

                    chunk = {
                        "chunk_id": chunk_id,
                        "parent_chunk_id": parent_id,
                        "content": chunk_content,
                        "metadata": chunk_meta,
                    }
                    all_chunks.append(chunk)

        return all_chunks
