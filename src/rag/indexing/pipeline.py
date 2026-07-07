"""索引管线 — 编排 7 阶段：load → clean → chunk → [refine] → [enrich] → embed → store."""

import hashlib
import os
import time

from src.config import settings
from src.observability.langfuse_context import langfuse_context

from src.rag.embeddings.text_embedding_v4 import TextEmbeddingV4
from src.rag.indexing.chunker import ParentChildChunker
from src.rag.indexing.metadata import extract_metadata
from src.rag.ingestion.cleaner import clean_text, compute_quality_score, deduplicate_sections, detect_injection, filter_short_chunks
from src.utils.exceptions import IngestionQualityError
from src.rag.ingestion.integrity import IngestionCache
from src.rag.ingestion.loader import load_document
from src.rag.ingestion.parser import parse_document
from src.rag.ingestion.refiner import ChunkRefiner
from src.rag.ingestion.enricher import MetadataEnricher
from src.rag.retrieval.collections import kb_collection_name
from src.stores.doc_store import DocStore
from src.utils.logger import logger
from src.observability.decorators import trace_rag_node


class IndexingPipeline:
    def __init__(self, doc_store: DocStore | None = None):
        self.embedder = TextEmbeddingV4()
        self.chunker = ParentChildChunker()
        self.doc_store = doc_store if doc_store is not None else DocStore()
        self._integrity = IngestionCache() if settings.ingestion_integrity_enabled else None

    @trace_rag_node(name="ingestion_pipeline")
    async def run(
        self, file_path: str, kb_id: str | None = None, title: str | None = None,
        force: bool = False,
    ) -> dict:
        t0 = time.monotonic()
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        ext = os.path.splitext(file_path)[1].lstrip(".").lower() or "txt"
        logger.info("pipeline_start", file=file_path)

        # ── Integrity check (SHA256 dedup) ──
        file_hash = _compute_file_hash(file_path)
        if self._integrity and not force:
            if self._integrity.should_skip(file_hash):
                logger.info("pipeline_skip_duplicate", file=file_path, hash=file_hash[:12])
                return {"file": file_path, "chunks": 0, "elapsed_ms": 0, "status": "skipped"}

        # ── Stage 1: Load ──
        stage_load_start = time.monotonic()
        try:
            raw_elements = load_document(file_path)
        except Exception:
            if self._integrity:
                self._integrity.mark_failed(file_hash, file_path, f"Load stage failed: {file_path}")
            raise
        stage_parse_load_ms = (time.monotonic() - stage_load_start) * 1000
        logger.info("pipeline_load", count=len(raw_elements), elapsed_ms=stage_parse_load_ms)

        # ── max_chunks_per_file enforcement (early truncation for structured data) ──
        _max_chunks = settings.ingestion_max_chunks_per_file
        if len(raw_elements) > _max_chunks:
            logger.warning(
                "max_chunks_truncated",
                file=file_path,
                total=len(raw_elements),
                limit=_max_chunks,
            )
            raw_elements = raw_elements[:_max_chunks]

        # ── Stage 2: Parse + Clean ──
        stage_parse_start = time.monotonic()
        try:
            parsed = parse_document(raw_elements)
            for p in parsed:
                p["content"] = clean_text(p["content"])
                flags = detect_injection(p["content"])
                if flags:
                    logger.warning("injection_detected", file=file_path, patterns=flags)
            parsed = [p for p in deduplicate_sections(parsed) if (p.get("content") or "").strip()]
            if not parsed:
                raise ValueError(
                    "No indexable text was found in the document. OCR may be required for scanned PDFs or image-only files."
                )
        except Exception:
            if self._integrity:
                self._integrity.mark_failed(file_hash, file_path, f"Parse/Clean stage failed: {file_path}")
            raise
        stage_parse_clean_ms = (time.monotonic() - stage_parse_start) * 1000
        logger.info("pipeline_clean", count=len(parsed), elapsed_ms=stage_parse_clean_ms)

        # ── Stage 2.5: Quality Check ──
        quality_score = 1.0
        if settings.ingestion_min_quality_score > 0:
            all_text = " ".join(p["content"] for p in parsed)
            quality_score = compute_quality_score(all_text)
            if quality_score < settings.ingestion_min_quality_score:
                raise IngestionQualityError(
                    f"文档质量不达标（得分 {quality_score:.2f}，要求 ≥ {settings.ingestion_min_quality_score:.2f}），"
                    f"存在大量乱码或无效字符，无法入库"
                )
            logger.info("pipeline_quality_check", score=quality_score, threshold=settings.ingestion_min_quality_score)

        # ── Stage 3: Chunk ──
        stage_chunk_start = time.monotonic()
        try:
            chunks = self.chunker.split_documents(parsed, doc_hash=file_hash)
        except Exception:
            if self._integrity:
                self._integrity.mark_failed(file_hash, file_path, f"Chunk stage failed: {file_path}")
            raise
        stage_chunk_ms = (time.monotonic() - stage_chunk_start) * 1000
        logger.info("pipeline_chunk", total_chunks=len(chunks), elapsed_ms=stage_chunk_ms)

        # ── Stage 3.2: Short Chunk Filter ──
        if settings.ingestion_min_chunk_length > 0:
            before = len(chunks)
            chunks = filter_short_chunks(chunks, settings.ingestion_min_chunk_length)
            removed = before - len(chunks)
            if removed:
                logger.info("pipeline_chunk_filter", removed=removed, kept=len(chunks),
                           min_length=settings.ingestion_min_chunk_length)

        # ── Stage 3.5: Refinement (optional) ──
        stage_refine_ms = 0.0
        if settings.rag_ingestion_chunk_refinement_enabled:
            stage_refine_start = time.monotonic()
            try:
                refiner = ChunkRefiner()
                chunks = await refiner.refine_chunks(chunks)
            except Exception:
                if self._integrity:
                    self._integrity.mark_failed(file_hash, file_path, f"Refinement stage failed: {file_path}")
                raise
            stage_refine_ms = (time.monotonic() - stage_refine_start) * 1000
            logger.info("pipeline_refine", total_chunks=len(chunks), elapsed_ms=stage_refine_ms)

        # ── Stage 3.7: Enrichment (optional) ──
        stage_enrich_ms = 0.0
        if settings.rag_ingestion_metadata_enrichment_enabled:
            stage_enrich_start = time.monotonic()
            try:
                enricher = MetadataEnricher()
                chunks = await enricher.enrich_chunks(chunks)
            except Exception:
                if self._integrity:
                    self._integrity.mark_failed(file_hash, file_path, f"Enrichment stage failed: {file_path}")
                raise
            stage_enrich_ms = (time.monotonic() - stage_enrich_start) * 1000
            logger.info("pipeline_enrich", total_chunks=len(chunks), elapsed_ms=stage_enrich_ms)

        # ── Stage 4: Embed ──
        stage_embed_start = time.monotonic()
        try:
            texts = [c["content"] for c in chunks]
            embeddings = await self.embedder.embed(texts)
        except Exception:
            if self._integrity:
                self._integrity.mark_failed(file_hash, file_path, f"Embed stage failed: {file_path}")
            raise
        stage_embed_ms = (time.monotonic() - stage_embed_start) * 1000
        logger.info("pipeline_embed", count=len(embeddings), elapsed_ms=stage_embed_ms)

        # ── Stage 5: Store (ChromaDB + DocStore) ──
        stage_store_start = time.monotonic()
        try:
            from src.rag.retrieval.dense import _get_vector_store
            doc_meta = extract_metadata(file_path, raw_elements)
            if title:
                doc_meta["source"] = title
            vs = await _get_vector_store()  # 复用持久连接
            col_name = kb_collection_name(str(kb_id)) if kb_id else "rag_docs_dev"
            col = await vs.get_or_create_collection(col_name)
            ids = [c["chunk_id"] for c in chunks]

            # Stage 5b: DocStore (PostgreSQL)
            stage_docstore_start = time.monotonic()
            doc_title = title if title else os.path.basename(file_path)
            doc_id = None
            try:
                doc_id = await self.doc_store.insert_document(
                    title=doc_title,
                    source=doc_title,
                    doc_type=ext,
                    file_hash=file_hash,
                    chunk_count=len(chunks),
                    metadata=doc_meta,
                    kb_id=kb_id,
                )
                logger.info("pipeline_docstore", doc_id=doc_id, elapsed_ms=(time.monotonic() - stage_docstore_start) * 1000)
            except Exception as e:
                logger.warning("docstore_write_failed", file=file_path, error=str(e))

            metadatas = [_sanitize_metadata({**c["metadata"], **doc_meta,
                           "doc_id": str(doc_id) if doc_id else "",
                           "kb_id": str(kb_id) if kb_id else "",
                           "chunk_id": c["chunk_id"],
                           "parent_chunk_id": c["parent_chunk_id"]}) for c in chunks]
            await vs.add(col, ids=ids, embeddings=embeddings,
                         metadatas=metadatas, documents=texts)
            # 标记 BM25 索引失效
            from src.graph.nodes.retrieval import invalidate_sparse_index
            await invalidate_sparse_index(col_name)
        except Exception:
            if self._integrity:
                self._integrity.mark_failed(file_hash, file_path, f"Store stage failed: {file_path}")
            raise
        stage_store_ms = (time.monotonic() - stage_store_start) * 1000
        logger.info("pipeline_store", count=len(ids), elapsed_ms=stage_store_ms)

        # ── Record integrity success ──
        if self._integrity:
            self._integrity.mark_success(
                file_hash, file_path,
                collection=col_name,
                chunk_count=len(chunks),
            )

        elapsed = (time.monotonic() - t0) * 1000

        # ── Record trace metadata ──
        langfuse_context.update_current_observation(
            node="ingestion_pipeline",
            file_name=file_name,
            file_size=file_size,
            doc_type=ext,
            chunk_count=len(chunks),
            kb_id=kb_id,
            persist_mode="persisted" if kb_id else "session_only",
            quality_score=quality_score,
            stage_parse_load_ms=round(stage_parse_load_ms, 2),
            stage_parse_clean_ms=round(stage_parse_clean_ms, 2),
            stage_chunk_ms=round(stage_chunk_ms, 2),
            stage_refine_ms=round(stage_refine_ms, 2),
            stage_enrich_ms=round(stage_enrich_ms, 2),
            stage_embed_ms=round(stage_embed_ms, 2),
            stage_store_ms=round(stage_store_ms, 2),
            total_ms=round(elapsed, 2),
            doc_id=doc_id,
        )

        logger.info("pipeline_done", file=file_path, chunks=len(chunks), total_ms=elapsed)
        return {"file": file_path, "chunks": len(chunks), "elapsed_ms": elapsed, "doc_id": doc_id, "status": "success"}


def _sanitize_metadata(meta: dict) -> dict:
    """清理 metadata，仅保留 ChromaDB 支持的类型（str, int, float, bool）。"""
    allowed = (str, int, float, bool)
    return {k: v for k, v in meta.items() if isinstance(v, allowed) and v is not None}


def _compute_file_hash(file_path: str) -> str:
    """计算文件内容的 SHA-256 哈希（十六进制字符串）。"""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()
