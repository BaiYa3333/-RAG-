"""RAG_Project FastAPI 应用入口.

lifespan 管理所有存储连接、LLM client、LangGraph workflow 的生命周期，
/health 端点检查基础设施健康状态，
/chat 和 /chat/stream 端点提供 RAG 查询接口。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.config import settings
from src.stores import VectorStore, DocStore, CacheStore
from src.utils.logger import logger
from src.utils.exceptions import RAGException


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 — 启动时初始化连接，关闭时清理资源。"""
    # ── 日志初始化 ────────────────────────────────
    from src.utils.logger import setup_logger as init_logger

    init_logger(settings.log_level)

    logger.info("app_starting", log_level=settings.log_level)

    # ── Startup ──────────────────────────────────
    app.state.vector_store = VectorStore()
    app.state.doc_store = DocStore()
    app.state.cache_store = CacheStore()

    try:
        await app.state.vector_store.connect()
    except Exception as e:
        logger.warning("chromadb_connect_failed", error=str(e))

    try:
        await app.state.doc_store.connect()
    except Exception as e:
        logger.warning(
            "postgres_connect_failed",
            error=str(e),
            detail="DocStore will run in degraded mode; API/database requests will report PostgreSQL is not connected.",
        )

    try:
        await app.state.cache_store.connect()
    except Exception as e:
        logger.warning("redis_connect_failed", error=str(e))

    # ── LLM Client ───────────────────────────────
    try:
        from src.llm.factory import create_llm

        app.state.llm_client = create_llm()
        logger.info("llm_client_created", model=app.state.llm_client.model_id)
    except Exception as e:
        logger.warning("llm_client_create_failed", error=str(e))
        app.state.llm_client = None

    # ── LangGraph Workflow ───────────────────────
    try:
        from src.graph.workflow import build_workflow

        app.state.graph = build_workflow()
        logger.info("graph_compiled")
    except Exception as e:
        logger.warning("graph_compile_failed", error=str(e))
        app.state.graph = None

    # ── IndexingPipeline ─────────────────────────
    try:
        from src.rag.indexing.pipeline import IndexingPipeline

        app.state.indexing_pipeline = IndexingPipeline(doc_store=app.state.doc_store)
        logger.info("indexing_pipeline_initialized")
    except Exception as e:
        logger.warning("indexing_pipeline_init_failed", error=str(e))
        app.state.indexing_pipeline = None

    # ── BM25 索引预热 ────────────────────────────
    try:
        from src.graph.nodes.retrieval import warmup_sparse_index

        await warmup_sparse_index()
    except Exception as e:
        logger.warning("bm25_warmup_failed", error=str(e))

    # ── RAGAS Evaluation Runner ──────────────────
    try:
        from src.evaluation.ragas_runner import EvaluationRunner

        app.state.evaluation_runner = EvaluationRunner(
            llm_client=app.state.llm_client,
            vector_store=app.state.vector_store,
            doc_store=app.state.doc_store,
        )
        logger.info("evaluation_runner_initialized")
    except Exception as e:
        logger.warning("evaluation_runner_init_failed", error=str(e))
        app.state.evaluation_runner = None

    # ── Auth Service ────────────────────────────
    try:
        from src.auth.service import AuthService

        app.state.auth_service = AuthService(app.state.doc_store)
        logger.info("auth_service_initialized")
    except Exception as e:
        logger.warning("auth_service_init_failed", error=str(e))
        app.state.auth_service = None

    # ── User Service ────────────────────────────
    try:
        from src.auth.user_service import UserService

        app.state.user_service = UserService(app.state.doc_store)
        logger.info("user_service_initialized")
    except Exception as e:
        logger.warning("user_service_init_failed", error=str(e))
        app.state.user_service = None

    # ── Memory Service ──────────────────────────
    try:
        from src.memory.service import MemoryService

        app.state.memory_service = MemoryService(app.state.doc_store)
        # 注入到 query_condenser 模块供记忆注入使用
        from src.graph.nodes.query_condenser import set_memory_service
        set_memory_service(app.state.memory_service)
        logger.info("memory_service_initialized")
    except Exception as e:
        logger.warning("memory_service_init_failed", error=str(e))
        app.state.memory_service = None

    # ── KnowledgeBase Service ───────────────────
    try:
        from src.rag.knowledge_base.service import KnowledgeBaseService

        app.state.kb_service = KnowledgeBaseService(app.state.doc_store)
        # 确保默认知识库存在
        await app.state.kb_service.ensure_default_kb()
        logger.info("kb_service_initialized")
    except Exception as e:
        logger.warning("kb_service_init_failed", error=str(e))
        app.state.kb_service = None

    # ── Langfuse Observability ──────────────────
    try:
        from src.observability.client import get_langfuse

        langfuse = get_langfuse()
        if langfuse is not None:
            logger.info("langfuse_observability_enabled")
        else:
            logger.info("langfuse_observability_disabled")
    except Exception as e:
        logger.warning("langfuse_observability_init_failed", error=str(e))

    # ── MCP Session Manager (Streamable HTTP) ───
    # 挂载本身在模块级完成（见文件底部）；session manager 必须在 lifespan 内运行
    mcp_session_ctx = None
    if settings.mcp_enabled:
        try:
            from src.mcp_server import service as mcp_service
            from src.mcp_server.server import mcp as mcp_instance

            # 注入共享 KB 服务，避免 MCP 层自建第二个 DocStore 连接池
            if app.state.kb_service is not None:
                mcp_service.set_kb_service(app.state.kb_service)

            mcp_session_ctx = mcp_instance.session_manager.run()
            await mcp_session_ctx.__aenter__()
            logger.info("mcp_http_enabled", path="/mcp")
        except Exception as e:
            logger.warning("mcp_session_manager_start_failed", error=str(e))
            mcp_session_ctx = None

    logger.info("app_started", host=settings.app_host, port=settings.app_port)
    yield
    # ── Shutdown ─────────────────────────────────
    # 先停 MCP session manager，避免关闭存储连接时仍有工具调用在途
    if mcp_session_ctx is not None:
        try:
            await mcp_session_ctx.__aexit__(None, None, None)
            logger.info("mcp_session_manager_stopped")
        except Exception as e:
            logger.warning("mcp_session_manager_stop_failed", error=str(e))

    # Flush Langfuse trace data before closing connections
    try:
        from src.observability.client import flush_langfuse
        flush_langfuse()
    except Exception as e:
        logger.warning("langfuse_flush_failed", error=str(e))

    # ── Shutdown ─────────────────────────────────
    # 关闭持久 VectorStore 连接（dense_search 复用的模块级连接）
    try:
        from src.rag.retrieval.dense import close_vector_store
        await close_vector_store()
    except Exception as e:
        logger.warning("dense_vector_store_close_failed", error=str(e))

    try:
        from src.rag.embeddings.text_embedding_v4 import close_embedding_cache
        await close_embedding_cache()
    except Exception as e:
        logger.warning("embedding_cache_close_failed", error=str(e))

    if app.state.cache_store:
        await app.state.cache_store.close()
    if app.state.doc_store:
        await app.state.doc_store.close()
    if app.state.vector_store:
        await app.state.vector_store.close()
    if app.state.llm_client:
        try:
            await app.state.llm_client.client.close()
        except Exception as e:
            logger.warning("llm_client_close_failed", error=str(e))
    logger.info("app_shutdown")


app = FastAPI(
    title="RAG_Project",
    description="企业级 RAG 系统 — 父子索引、混合检索、RRF 融合、LangGraph 编排",
    version="0.1.0",
    lifespan=lifespan,
)

# ── 异常处理 ─────────────────────────────────────
from src.api.errors import rag_exception_handler, generic_exception_handler

app.add_exception_handler(RAGException, rag_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# ── 中间件 ───────────────────────────────────────
from src.api.middleware import RateLimitMiddleware

app.add_middleware(RateLimitMiddleware)

# ── 路由 ─────────────────────────────────────────
from src.api.routes import router as chat_router
from src.api.routes import documents_router, models_router
from src.api.evaluation_routes import evaluation_router
from src.api.admin_routes import admin_router
from src.api.session_routes import sessions_router, memory_router
from src.api.auth_routes import auth_router
from src.api.kb_routes import kb_router

app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(models_router)
app.include_router(evaluation_router)
app.include_router(admin_router)
app.include_router(sessions_router)
app.include_router(auth_router)
app.include_router(memory_router)
app.include_router(kb_router)

# ── MCP Streamable HTTP 注册（默认关闭）─────────
# 以精确 Route("/mcp") 注册 ASGI 守卫（与 FastMCP 内部注册方式一致）。
# 不用 app.mount：Mount 前缀匹配对不带尾斜杠的 /mcp 会先 307 重定向，
# API Key 守卫不执行且不跟随重定向的严格 MCP 客户端会失败。
# config 已保证 enabled 时 key ≥16 字符。
if settings.mcp_enabled:
    from starlette.routing import Route

    from src.mcp_server.auth import MCPAPIKeyGuard
    from src.mcp_server.server import mcp as _mcp_instance

    # streamable_http_app 的内部路由为 /mcp（FastMCP 默认），Route 精确匹配后
    # 原样透传 scope.path="/mcp"，两者一致，无 /mcp/mcp 双前缀
    app.router.routes.append(
        Route(
            "/mcp",
            endpoint=MCPAPIKeyGuard(_mcp_instance.streamable_http_app(), settings.mcp_api_key),
        )
    )


@app.get("/health")
async def health():
    """检查所有基础设施服务和 RAG 组件的连接状态。"""
    services = {}

    # PostgreSQL
    try:
        if app.state.doc_store and app.state.doc_store._pool:
            await app.state.doc_store.health()
            services["postgres"] = "connected"
        else:
            services["postgres"] = "not_connected: PostgreSQL is not connected. Check RAG_POSTGRES_DSN and startup logs."
    except Exception as e:
        services["postgres"] = f"error: {e}"

    # Redis
    try:
        if app.state.cache_store and app.state.cache_store._redis:
            await app.state.cache_store.health()
            services["redis"] = "connected"
        else:
            services["redis"] = "not_initialized"
    except Exception as e:
        services["redis"] = f"error: {e}"

    # ChromaDB
    try:
        if app.state.vector_store and app.state.vector_store._client:
            await app.state.vector_store.heartbeat()
            services["chromadb"] = "connected"
        else:
            services["chromadb"] = "not_initialized"
    except Exception as e:
        services["chromadb"] = f"error: {e}"

    # LLM Client
    services["llm"] = "ready" if app.state.llm_client is not None else "not_initialized"

    # Graph
    services["graph"] = "ready" if app.state.graph is not None else "not_initialized"

    # IndexingPipeline
    pipeline = getattr(app.state, "indexing_pipeline", None)
    services["indexing_pipeline"] = "ready" if pipeline is not None else "not_initialized"

    # RAGAS Evaluation
    eval_runner = getattr(app.state, "evaluation_runner", None)
    if eval_runner is not None:
        try:
            import importlib
            importlib.import_module("ragas")
            services["evaluation"] = "ready"
        except Exception:
            # ragas 依赖兼容性问题（如 langchain_community.chat_models.vertexai 路径变更）
            # evaluation_runner 已初始化，标记为 degraded 而非 not_initialized
            services["evaluation"] = "degraded"
    else:
        services["evaluation"] = "not_initialized"

    # 汇总状态 — status values starting with "not_connected" or "error" report clear diagnostics
    _healthy_prefixes = ("connected", "not_initialized", "ready", "degraded", "not_connected")
    all_healthy = all(
        any(v.startswith(prefix) for prefix in _healthy_prefixes) for v in services.values()
    )
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        content={"status": "healthy" if all_healthy else "degraded", "services": services},
        status_code=status_code,
    )


# ── 静态页面 ──────────────────────────────────
from pathlib import Path
from fastapi.responses import FileResponse, RedirectResponse

from src.auth.user_service import decode_jwt

UI_DIR = Path(__file__).parent / "ui"
LOGIN_HTML = UI_DIR / "login.html"
INDEX_HTML = UI_DIR / "index.html"


@app.get("/")
async def serve_login():
    """返回登录/注册页面."""
    return FileResponse(LOGIN_HTML)


@app.get("/app")
async def serve_app(request: Request):
    """返回聊天界面 HTML 页面（需登录）."""
    # Check JWT cookie for authentication
    token = request.cookies.get("rag_jwt")
    if not token:
        return RedirectResponse(url="/", status_code=302)

    payload = decode_jwt(token)
    if not payload:
        # Clear invalid cookie and redirect
        response = RedirectResponse(url="/", status_code=302)
        response.delete_cookie("rag_jwt", path="/")
        return response

    return FileResponse(INDEX_HTML)


@app.get("/eval")
async def eval_redirect(request: Request):
    """便捷跳转：/eval → /eval/dashboard（需登录）."""
    token = request.cookies.get("rag_jwt")
    if not token or not decode_jwt(token):
        return RedirectResponse(url="/", status_code=302)
    return RedirectResponse(url="/eval/dashboard", status_code=302)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host=settings.app_host, port=settings.app_port, reload=True)
