"""API 路由 — POST /chat (非流式) + POST /chat/stream (SSE 流式) + POST /documents/upload (文档上传)."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

from fastapi import APIRouter, Depends, Form, Request, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.observability.langfuse_context import langfuse_context

from src.api.schemas import ChatRequest
from src.auth.dependencies import get_current_user
from src.config import settings
from src.graph.workflow import run_query, run_query_stream
from src.rag.generation.generator import generate_stream
from src.llm.registry import MODEL_REGISTRY
from src.rag.retrieval.collections import session_collection_name
from src.observability.decorators import trace_rag_node
from src.api.sse_events import (
    SSEEventType,
    intent_event,
    retrieval_chunk_event,
    quality_gate_event,
    rerank_event,
    token_event,
    done_event,
    error_event,
    parse_start_event,
    parse_complete_event,
    chunking_start_event,
    chunking_complete_event,
    embedding_start_event,
    embedding_complete_event,
    ingestion_done_event,
)
from src.utils.logger import logger
from src.utils.exceptions import IngestionQualityError

router = APIRouter(prefix="/chat", tags=["chat"])
documents_router = APIRouter(prefix="/documents", tags=["documents"])
models_router = APIRouter(prefix="/models", tags=["models"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".markdown", ".html", ".htm", ".txt", ".rst"}


def _build_response(state: dict) -> dict:
    """将 RAGState dict 映射为 API 响应格式."""
    docs = []
    for doc in state.get("documents") or []:
        docs.append({
            "content": doc.get("content", ""),
            "score": doc.get("score", doc.get("rrf_score", 0.0)),
            "metadata": doc.get("metadata", {}),
        })

    gate_log = state.get("gate_log") or []
    gate_entries = []
    for entry in gate_log:
        gate_entries.append({
            "tier": entry.get("tier", 0),
            "action": entry.get("action", ""),
            "avg_score": entry.get("avg_score"),
            "reason": entry.get("reason"),
            "threshold": entry.get("threshold"),
            "doc_count": entry.get("doc_count"),
        })

    return {
        "query": state.get("query", ""),
        "answer": state.get("answer", ""),
        "intent": state.get("intent", ""),
        "documents": docs,
        "retrieval_tier": state.get("retrieval_tier", 1),
        "gate_log": gate_entries,
        "errors": state.get("errors") or [],
        "model": state.get("model_name", settings.default_llm_model),
        "session_id": state.get("session_id"),
        "citations": state.get("citations", {}),
    }


def _validate_model_name(model: str | None) -> str | None:
    if model and model not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise HTTPException(status_code=422, detail=f"Unknown model: {model}. Available: {available}")
    return model


async def _resolve_kb_ids(request: Request, kb_ids: list[str] | None, user_id: str | None) -> list[str] | None:
    """当 kb_ids 为空时，自动解析为用户可访问的所有知识库。

    如果用户未选择特定知识库，则将其所有可访问的知识库纳入检索范围，
    避免回退到空的 rag_docs_dev 遗留集合导致检索无结果。
    """
    if kb_ids:
        return kb_ids

    kb_service = getattr(request.app.state, "kb_service", None)
    if kb_service is None:
        return None  # KB service not available, fall back to default collection

    try:
        accessible_kbs = await kb_service.list_kbs(user_id=user_id)
        resolved = [kb["id"] for kb in accessible_kbs]
        if resolved:
            logger.info("kb_ids_auto_resolved", count=len(resolved), user_id=user_id)
            return resolved
    except Exception as e:
        logger.warning("kb_ids_resolve_failed", error=str(e))

    return None  # No KBs accessible, fall back to default collection


@models_router.get("", summary="列出可用 LLM 模型")
async def list_models():
    """返回模型注册表中的可用模型，供 Web UI 和客户端选择。"""
    from src.llm.registry import DEFAULT_MODEL, MODEL_REGISTRY

    models = [
        {
            "name": name,
            "provider": spec.provider,
            "label": spec.display_label() if hasattr(spec, "display_label") else name,
            "default": name == DEFAULT_MODEL,
        }
        for name, spec in MODEL_REGISTRY.items()
    ]
    return {"models": models, "default": DEFAULT_MODEL}


@router.post("", summary="非流式 RAG 查询")
@trace_rag_node(name="api_chat")
async def chat(req: ChatRequest, request: Request, current_user: dict = Depends(get_current_user)):
    """提交 RAG 查询，返回完整答案及来源.

    当 session_id 存在时，自动加载持久历史并追加到 chat_history，
    查询完成后将本轮对话持久化。
    """
    graph = getattr(request.app.state, "graph", None)
    memory_service = getattr(request.app.state, "memory_service", None)

    # Verify session ownership when session_id is provided
    if req.session_id:
        await _verify_session_ownership(req.session_id, request)

    # Merge persisted history with request chat_history
    chat_history = list(req.chat_history or [])
    if req.session_id and memory_service:
        try:
            persisted = await memory_service.load_history(req.session_id)
            # Prepend persisted history (older) before request history
            chat_history = persisted + chat_history
        except Exception as e:
            logger.warning("memory_load_failed", session_id=req.session_id, error=str(e))

    # 模型名校验在 try 外执行，避免 HTTPException(422) 被误吞为 503
    model = _validate_model_name(req.model)

    try:
        # Extract user_id from auth context
        user_id = req.user_id or current_user.get("user_id") or "anonymous"

        # Set trace-level metadata
        langfuse_context.update_current_trace(
            session_id=req.session_id,
            user_id=user_id,
            query=req.query,
            model=model,
            language=req.language or "zh",
        )

        # Auto-resolve kb_ids when empty (search all accessible KBs)
        resolved_kb_ids = await _resolve_kb_ids(request, req.kb_ids, user_id)

        state = await run_query(
            query=req.query,
            chat_history=chat_history,
            graph=graph,
            model=model,
            session_id=req.session_id,
            language=req.language,
            user_id=user_id,
            kb_ids=resolved_kb_ids,
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error("chat_pipeline_failed", error=str(e), traceback=tb)
        raise HTTPException(status_code=503, detail=f"RAG pipeline unavailable: {e}")

    # Persist conversation turn
    if req.session_id and memory_service:
        try:
            await memory_service.append_turn(req.session_id, "user", req.query)
            await memory_service.append_turn(req.session_id, "assistant", state.get("answer", ""))
        except Exception as e:
            logger.warning("memory_persist_failed", session_id=req.session_id, error=str(e))

    # 异步提取用户长期记忆（非阻塞）
    user_id = state.get("user_id", "")
    if user_id and user_id != "anonymous" and memory_service:
        try:
            asyncio.create_task(
                _extract_memories_background(
                    query=req.query,
                    answer=state.get("answer", ""),
                    user_id=user_id,
                    memory_service=memory_service,
                    session_id=req.session_id,
                )
            )
        except Exception as e:
            logger.warning("memory_extract_task_failed", error=str(e))

    return _build_response(state)


@router.post("/stream", summary="流式 RAG 查询 (SSE)")
async def chat_stream(req: ChatRequest, request: Request, current_user: dict = Depends(get_current_user)):
    """提交 RAG 查询，通过 SSE 逐事件返回结果。

    事件类型: intent | retrieval_chunk | quality_gate | rerank | token | done | error
    当 session_id 存在时，自动加载持久历史并持久化本轮对话。

    trace 追踪：使用 @trace_rag_node(name="api_chat") 装饰内部 async generator，
    确保 trace 覆盖整个流式请求生命周期（而非仅 chat_stream 函数返回时）。
    """
    graph = getattr(request.app.state, "graph", None)
    memory_service = getattr(request.app.state, "memory_service", None)
    model = _validate_model_name(req.model)

    # Verify session ownership when session_id is provided
    if req.session_id:
        await _verify_session_ownership(req.session_id, request)

    # Merge persisted history (prep-work before the traced generator)
    chat_history = list(req.chat_history or [])
    if req.session_id and memory_service:
        try:
            persisted = await memory_service.load_history(req.session_id)
            chat_history = persisted + chat_history
            # Persist user query immediately
            try:
                await memory_service.append_turn(req.session_id, "user", req.query)
            except Exception as e:
                logger.warning("memory_persist_user_failed", error=str(e))
        except Exception as e:
            logger.warning("memory_load_failed", session_id=req.session_id, error=str(e))

    # Extract user_id from auth context
    stream_user_id = req.user_id or current_user.get("user_id") or "anonymous"

    # Auto-resolve kb_ids when empty (search all accessible KBs)
    resolved_kb_ids = await _resolve_kb_ids(request, req.kb_ids, stream_user_id)

    # ── Traced event generator ──────────────────────────────────
    # @trace_rag_node wraps the async generator: Langfuse v2 @observe()
    # starts the span on first __anext__ and ends it when the generator
    # is exhausted, covering the full streaming lifecycle.
    @trace_rag_node(name="api_chat")
    async def event_generator():
        errors: list[str] = []
        streamed_intent = ""
        streamed_tier = 1
        streamed_docs_count = 0
        streamed_answer = ""
        streamed_citations: dict = {}
        streamed_documents: list[dict] = []

        # Set trace-level metadata (now inside the @observe() context)
        langfuse_context.update_current_trace(
            session_id=req.session_id,
            user_id=stream_user_id,
            query=req.query,
            model=model,
            language=req.language or "zh",
        )

        try:
            async for chunk in run_query_stream(
                query=req.query,
                chat_history=chat_history,
                graph=graph,
                model=model,
                session_id=req.session_id,
                language=req.language,
                user_id=stream_user_id,
                kb_ids=resolved_kb_ids,
            ):
                if not isinstance(chunk, dict):
                    continue

                if "error" in chunk:
                    errors.append(chunk["error"])
                    yield f"event: error\ndata: {json.dumps(error_event(message=chunk['error']), ensure_ascii=False)}\n\n"
                    continue

                for node_name, node_state in chunk.items():
                    # Handle generate_answer in stream mode — use generate_stream for true token streaming
                    if node_name == "generate_answer" and node_state.get("streaming_context"):
                        streaming_context = node_state.get("streaming_context", "")
                        streaming_intent = node_state.get("streaming_intent", streamed_intent)
                        citations = node_state.get("citations", {})
                        if citations:
                            streamed_citations = citations

                        if streaming_context:
                            try:
                                async for token in generate_stream(
                                    query=req.query,
                                    context=streaming_context,
                                    model_name=model,
                                    language=req.language,
                                    format_mode="auto",
                                    intent=streaming_intent,
                                ):
                                    streamed_answer += token
                                    yield f"event: token\ndata: {json.dumps(token_event(token=token), ensure_ascii=False)}\n\n"
                            except Exception as gen_exc:
                                logger.error("stream_generate_failed", error=str(gen_exc))
                                errors.append(str(gen_exc))
                                yield f"event: error\ndata: {json.dumps(error_event(message=str(gen_exc)), ensure_ascii=False)}\n\n"
                    else:
                        event_type, payload = _node_to_event(node_name, node_state)
                        if event_type and payload:
                            if event_type == "intent":
                                streamed_intent = payload.get("intent", "")
                            elif event_type == "documents":
                                streamed_tier = payload.get("tier", 1)
                                streamed_docs_count = len(payload.get("documents", []))
                            elif event_type == "token":
                                # Non-stream-mode token (fallback path)
                                streamed_answer += payload.get("token", "")
                            yield f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

                    # Capture citations from generate_answer node
                    if node_name == "generate_answer":
                        citations = node_state.get("citations", {})
                        if citations:
                            streamed_citations = citations
                        # 保存文档列表供后续引用解析（优先用完整 docs）
                        docs = node_state.get("documents") or []
                        if docs:
                            streamed_documents = docs

                    # 保存检索结果文档（引用解析需要完整文档含 metadata）
                    if node_name in ("tier1_retrieve", "tier2_retrieve"):
                        docs = node_state.get("documents") or []
                        if docs:
                            streamed_documents = docs

            # Parse citations from streamed answer after streaming completes
            if streamed_answer and not streamed_citations and streamed_documents:
                try:
                    from src.rag.generation.generator import parse_citations
                    streamed_citations = parse_citations(streamed_answer, streamed_documents)
                except Exception as e:
                    logger.warning("stream_citation_parse_failed", error=str(e))

            # Persist assistant answer to memory
            if req.session_id and memory_service and streamed_answer:
                try:
                    await memory_service.append_turn(req.session_id, "assistant", streamed_answer)
                except Exception as e:
                    logger.warning("memory_persist_stream_failed", error=str(e))

            # Extract long-term memories in background (non-blocking)
            if stream_user_id and stream_user_id != "anonymous" and memory_service and streamed_answer:
                try:
                    asyncio.create_task(
                        _extract_memories_background(
                            query=req.query,
                            answer=streamed_answer,
                            user_id=stream_user_id,
                            memory_service=memory_service,
                            session_id=req.session_id,
                        )
                    )
                except Exception as e:
                    logger.warning("stream_memory_extract_task_failed", error=str(e))

            # Done event with full summary
            done_data = done_event(
                intent=streamed_intent,
                retrieval_tier=streamed_tier,
                documents_count=streamed_docs_count,
                model=model or settings.default_llm_model,
                session_id=req.session_id,
                citations=streamed_citations,
            )
            yield f"event: done\ndata: {json.dumps(done_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error("chat_stream_failed", error=str(e))
            errors.append(str(e))
            yield f"event: error\ndata: {json.dumps(error_event(message=str(e)), ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@documents_router.post("/upload", summary="上传文档并触发摄入")
@trace_rag_node(name="api_document_upload")
async def upload_document(
    file: UploadFile,
    request: Request,
    kb_id: str = Form(default=None),
    session_id: str = Form(default=None),
    current_user: dict = Depends(get_current_user),
):
    """接收 multipart 文件上传，调用 IndexingPipeline 完成全链路摄入。

    可选参数 kb_id 指定目标知识库（支持 query string 或 form field）。
    可选参数 session_id 指定会话 ID（session-only 上传时使用）。
    未指定知识库时：所有用户上传仅会话内有效，不持久化。
    指定知识库时：通过 KB 权限检查后持久化到对应知识库的向量库和数据库。
    返回文档元数据与分块统计。
    """
    # ── 校验扩展名 ─────────────────────────────────
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # ── 校验非空 ───────────────────────────────────
    if file.size is not None and file.size == 0:
        raise HTTPException(status_code=422, detail="File must not be empty")

    # ── 校验大小 ───────────────────────────────────
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content_length = request.headers.get("Content-Length")
    if content_length is not None and int(content_length) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum is {settings.max_upload_size_mb} MB",
        )

    # ── 临时文件落地 ────────────────────────────────
    tmp_path = None
    try:
        # 保留原始扩展名，unstructured 依赖扩展名选择解析器
        suffix = ext if ext else ".tmp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            chunk_data = await file.read(65536)  # 64KB 流式读取
            while chunk_data:
                tmp.write(chunk_data)
                chunk_data = await file.read(65536)

        # 二次大小检查（流式读取后确认实际大小）
        actual_size = os.path.getsize(tmp_path)
        if actual_size == 0:
            raise HTTPException(status_code=422, detail="File must not be empty")
        if actual_size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum is {settings.max_upload_size_mb} MB",
            )

        # ── 未指定知识库：仅会话内有效，不持久化 ─────────
        if not kb_id:
            if session_id:
                await _verify_session_ownership(session_id, request)
            return await _upload_session_only(tmp_path, file.filename or "unknown", ext, kb_id, request, session_id)

        # ── 检查目标知识库的写入权限 ───────────────────
        user_id = current_user.get("user_id")
        kb_service = getattr(request.app.state, "kb_service", None)
        if kb_service and user_id:
            has_permission = await kb_service.check_permission(kb_id, user_id, required="write")
            if not has_permission:
                raise HTTPException(
                    status_code=403,
                    detail="No write permission on this knowledge base",
                )
        else:
            # No kb_service or no user_id — fall back to session-only
            if session_id:
                await _verify_session_ownership(session_id, request)
            return await _upload_session_only(tmp_path, file.filename or "unknown", ext, kb_id, request, session_id)
        pipeline = getattr(request.app.state, "indexing_pipeline", None)
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Indexing pipeline not initialized")

        # ── 执行摄入 ────────────────────────────────
        original_filename = file.filename or "unknown"
        result = await pipeline.run(tmp_path, kb_id=kb_id, title=original_filename)

        # ── 计算文件哈希（从 pipeline result 取；若未取到则现场计算） ─
        from src.rag.indexing.pipeline import _compute_file_hash

        file_hash = _compute_file_hash(tmp_path)

        return {
            "doc_id": result.get("doc_id"),
            "title": os.path.basename(file.filename or "unknown"),
            "source": file.filename or "unknown",
            "doc_type": ext.lstrip(".") or "unknown",
            "chunk_count": result.get("chunks", 0),
            "status": result.get("status", "indexed" if result.get("doc_id") else "partial"),
            "file_hash": file_hash,
            "metadata": {},
            "persist_mode": "persisted",
        }

    except IngestionQualityError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error("upload_ingestion_failed", file=file.filename, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {e}",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def _upload_session_only(
    tmp_path: str,
    filename: str,
    ext: str,
    kb_id: str | None,
    request: Request,
    session_id: str | None = None,
) -> dict:
    """普通用户上传：解析+分块+向量化，写入 ChromaDB 会话级集合 rag_session_{session_id}。

    会话级上传的文档仅在该会话内可检索，不持久化到数据库或全局向量库。
    如果未提供 session_id，自动创建一个新会话。
    """
    from src.rag.ingestion.loader import load_document
    from src.rag.ingestion.parser import parse_document
    from src.rag.ingestion.cleaner import clean_text, compute_quality_score, deduplicate_sections, filter_short_chunks
    from src.rag.indexing.chunker import ParentChildChunker
    from src.rag.embeddings.text_embedding_v4 import TextEmbeddingV4
    from src.rag.indexing.pipeline import _compute_file_hash, _sanitize_metadata
    from src.rag.retrieval.dense import _get_vector_store

    # Auto-create session if no session_id provided
    if not session_id:
        memory_service = getattr(request.app.state, "memory_service", None)
        if memory_service:
            try:
                user_id = None
                try:
                    auth_header = request.headers.get("Authorization", "")
                    if auth_header.startswith("Bearer "):
                        token = auth_header[7:]
                        from src.auth.user_service import decode_jwt
                        payload = decode_jwt(token)
                        if payload:
                            user_id = payload.get("sub")
                except Exception:
                    pass
                session = await memory_service.create_session(user_id=user_id, title="上传文档")
                session_id = session["id"]
            except Exception as e:
                logger.warning("auto_create_session_for_upload_failed", error=str(e))
                # Fallback: generate a UUID-based session ID that won't be in DB
                import uuid
                session_id = str(uuid.uuid4())

    # Stage 1-2: Load + Parse + Clean
    raw_elements = load_document(tmp_path)
    parsed = parse_document(raw_elements)
    for p in parsed:
        p["content"] = clean_text(p["content"])
    parsed = deduplicate_sections(parsed)

    # Quality check (same as pipeline Stage 2.5)
    if settings.ingestion_min_quality_score > 0:
        all_text = " ".join(p.get("content", "") for p in parsed)
        quality = compute_quality_score(all_text)
        if quality < settings.ingestion_min_quality_score:
            raise IngestionQualityError(
                f"文档质量不达标（得分 {quality:.2f}，要求 ≥ {settings.ingestion_min_quality_score:.2f}），"
                f"存在大量乱码或无效字符，无法入库"
            )

    # Stage 3: Chunk — reuse pipeline chunker when available
    pipeline = getattr(request.app.state, "indexing_pipeline", None)
    pipeline_chunker = getattr(pipeline, "chunker", None) if pipeline else None
    chunker = pipeline_chunker if isinstance(pipeline_chunker, ParentChildChunker) else ParentChildChunker()
    chunks = chunker.split_documents(parsed)

    # Short chunk filter (same as pipeline Stage 3.2)
    if settings.ingestion_min_chunk_length > 0:
        before = len(chunks)
        chunks = filter_short_chunks(chunks, settings.ingestion_min_chunk_length)
        if before != len(chunks):
            logger.info("session_upload_chunk_filter", removed=before - len(chunks),
                       kept=len(chunks))

    # Stage 4: Embed — reuse pipeline embedder when available
    pipeline_embedder = getattr(pipeline, "embedder", None) if pipeline else None
    embedder = pipeline_embedder if isinstance(pipeline_embedder, TextEmbeddingV4) else TextEmbeddingV4()
    texts = [c["content"] for c in chunks]
    embeddings = await embedder.embed(texts)

    # Stage 5: Write to ChromaDB session collection
    try:
        vs = await _get_vector_store()
        col_name = session_collection_name(str(session_id))
        col = await vs.get_or_create_collection(col_name)

        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {**c.get("metadata", {}), "source": filename, "session_id": session_id}
            for c in chunks
        ]
        # Store child content as document, parent_content in metadata
        documents = [c["content"] for c in chunks]
        for i, c in enumerate(chunks):
            if c.get("parent_content"):
                metadatas[i]["parent_content"] = c["parent_content"]

        # Sanitize metadata to remove non-primitive values (lists, dicts)
        # that would cause ChromaDB ValueError
        metadatas = [_sanitize_metadata(m) for m in metadatas]

        await vs.add(col, ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        logger.info(
            "session_upload_stored",
            session_id=session_id,
            collection=col_name,
            chunks=len(chunks),
            file=filename,
        )
    except Exception as e:
        logger.error("session_upload_chromadb_failed", error=str(e), session_id=session_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store document in session collection: {e}",
        )

    file_hash = _compute_file_hash(tmp_path)
    title = os.path.basename(filename)

    return {
        "doc_id": None,
        "title": title,
        "source": filename,
        "doc_type": ext.lstrip(".") or "unknown",
        "chunk_count": len(chunks),
        "status": "session_only",
        "file_hash": file_hash,
        "metadata": {},
        "persist_mode": "session_only",
        "session_id": session_id,
    }


def _node_to_event(node_name: str, state: dict) -> tuple[str | None, dict | None]:
    """将 LangGraph node chunk 映射为 SSE event (type, payload)."""

    # Intent Router 输出
    if node_name == "intent_router":
        return ("intent", {
            "intent": state.get("intent", ""),
            "confidence": state.get("router_confidence", 0.0),
        })

    # 检索完成
    if node_name in ("tier1_retrieve", "tier2_retrieve"):
        docs = state.get("documents") or []
        return ("documents", {
            "documents": [
                {"content": d.get("content", "")[:200], "score": d.get("score", d.get("rrf_score", 0.0))}
                for d in docs[:5]
            ],
            "tier": 2 if node_name == "tier2_retrieve" else 1,
        })

    # 生成 token（含 chitchat_answer / generate_answer 的非流式 fallback）
    if node_name in ("generate_answer", "chitchat_answer"):
        answer = state.get("answer", "")
        return ("token", {"token": answer}) if answer else (None, None)

    # 质量门控
    if node_name == "quality_gate":
        gate_log = state.get("gate_log") or []
        last = gate_log[-1] if gate_log else {}
        return ("gate", {"tier": last.get("tier", 0), "action": last.get("action", "")})

    return (None, None)


@documents_router.post("/upload/stream", summary="Streaming document upload (SSE)")
async def upload_document_stream(file: UploadFile, request: Request, current_user: dict = Depends(get_current_user)):
    """Upload a document and stream ingestion progress via SSE.

    Events: parse_start | parse_complete | chunking_start | chunking_complete
           | embedding_complete | ingestion_done | error
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content_length = request.headers.get("Content-Length")
    if content_length is not None and int(content_length) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum is {settings.max_upload_size_mb} MB")

    async def event_generator():
        tmp_path = None
        try:
            yield f"event: parse_start\ndata: {json.dumps(parse_start_event(file.filename or 'unknown'), ensure_ascii=False)}\n\n"

            suffix = ext if ext else ".tmp"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
                chunk = await file.read(65536)
                while chunk:
                    tmp.write(chunk)
                    chunk = await file.read(65536)

            actual_size = os.path.getsize(tmp_path)
            if actual_size == 0:
                yield f"event: error\ndata: {json.dumps(error_event(message='Empty file'), ensure_ascii=False)}\n\n"
                return
            if actual_size > max_bytes:
                yield f"event: error\ndata: {json.dumps(error_event(message='File too large'), ensure_ascii=False)}\n\n"
                return

            doc_type = ext.lstrip(".") or "unknown"
            yield f"event: parse_complete\ndata: {json.dumps(parse_complete_event(file.filename or 'unknown', doc_type), ensure_ascii=False)}\n\n"
            yield f"event: chunking_start\ndata: {json.dumps(chunking_start_event(settings.chunk_size), ensure_ascii=False)}\n\n"

            pipeline = getattr(request.app.state, "indexing_pipeline", None)
            if pipeline is None:
                yield f"event: error\ndata: {json.dumps(error_event(message='Pipeline not available'), ensure_ascii=False)}\n\n"
                return

            result = await pipeline.run(tmp_path)
            chunk_count = result.get("chunks", 0)
            yield f"event: chunking_complete\ndata: {json.dumps(chunking_complete_event(chunk_count), ensure_ascii=False)}\n\n"
            yield f"event: embedding_complete\ndata: {json.dumps(embedding_complete_event(chunk_count), ensure_ascii=False)}\n\n"
            done_event = ingestion_done_event(
                doc_id=result.get('doc_id', ''),
                chunk_count=chunk_count,
                title=os.path.basename(file.filename or 'unknown'),
            )
            yield f"event: ingestion_done\ndata: {json.dumps(done_event, ensure_ascii=False)}\n\n"

        except HTTPException:
            raise
        except Exception as e:
            logger.error("upload_stream_failed", file=file.filename, error=str(e))
            yield f"event: error\ndata: {json.dumps(error_event(message=str(e)), ensure_ascii=False)}\n\n"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _extract_memories_background(
    query: str,
    answer: str,
    user_id: str,
    memory_service,
    session_id: str | None = None,
) -> None:
    """后台任务：从本轮对话中提取长期记忆."""
    try:
        from src.memory.extractor import extract_and_save_memories
        conversation = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
        await extract_and_save_memories(
            conversation, user_id, memory_service, session_id=session_id,
        )
    except Exception as e:
        logger.warning("background_memory_extract_failed", error=str(e))


# ── Session & Memory routes (extracted to session_routes.py) ──
from src.api.session_routes import sessions_router, memory_router, _verify_session_ownership
