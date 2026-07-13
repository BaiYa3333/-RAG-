"""RAG_Project 全局配置 — Pydantic Settings.

所有配置项通过环境变量注入，命名前缀 RAG_。
环境变量从 .env 文件加载。
"""

import os
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Unsafe secret constants (module-level for external tooling & test access) ──
UNSAFE_JWT_SECRETS: frozenset[str] = frozenset({
    "change-me-in-production-use-random-secret",
    "changeme",
    "change-me",
    "secret",
})
UNSAFE_ADMIN_CODES: frozenset[str] = frozenset({
    "321458",
    "changeme",
    "change-me",
    "admin",
})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    # ── LLM ──────────────────────────────────────────────────
    default_llm_model: str = Field(
        default="deepseek-chat",
        alias="RAG_DEFAULT_LLM_MODEL",
        description="默认 LLM 模型名 (registry key)",
    )
    deepseek_api_key: str = Field(
        default="sk-xxx",
        alias="RAG_DEEPSEEK_API_KEY",
        description="DeepSeek API Key",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        alias="RAG_DEEPSEEK_BASE_URL",
    )
    qwen_api_key: str = Field(
        default="sk-xxx",
        alias="RAG_QWEN_API_KEY",
        description="Qwen (DashScope) API Key",
    )
    qwen_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="RAG_QWEN_BASE_URL",
    )

    # ── Embedding ────────────────────────────────────────────
    embedding_model: str = Field(
        default="text-embedding-v4",
        alias="RAG_EMBEDDING_MODEL",
    )
    embedding_api_key: str = Field(
        default="sk-xxx",
        alias="RAG_EMBEDDING_API_KEY",
    )
    embedding_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="RAG_EMBEDDING_BASE_URL",
    )
    embedding_batch_size: int = Field(
        default=10,
        alias="RAG_EMBEDDING_BATCH_SIZE",
        description="Embedding API 单次最大批量（DashScope 上限 10）",
    )

    # ── ChromaDB ─────────────────────────────────────────────
    chroma_host: str = Field(
        default="localhost",
        alias="RAG_CHROMA_HOST",
    )
    chroma_port: int = Field(
        default=8000,
        alias="RAG_CHROMA_PORT",
    )

    # ── PostgreSQL ───────────────────────────────────────────
    postgres_dsn: str = Field(
        default="postgresql://raguser:changeme@localhost:5432/ragdb",
        alias="RAG_POSTGRES_DSN",
    )
    postgres_min_pool: int = Field(default=2, alias="RAG_POSTGRES_MIN_POOL")
    postgres_max_pool: int = Field(default=10, alias="RAG_POSTGRES_MAX_POOL")

    # ── Redis ────────────────────────────────────────────────
    redis_host: str = Field(default="localhost", alias="RAG_REDIS_HOST")
    redis_port: int = Field(default=6379, alias="RAG_REDIS_PORT")
    redis_db: int = Field(default=0, alias="RAG_REDIS_DB")

    # ── RAG 参数 ─────────────────────────────────────────────
    tier1_score_threshold: float = Field(
        default=0.35,
        alias="RAG_TIER1_SCORE_THRESHOLD",
        description="Tier1 quality gate 向量相似度阈值（cosine，值域 0~1）",
    )
    chunk_size: int = Field(
        default=256,
        alias="RAG_CHUNK_SIZE",
        description="子块字符数（RecursiveCharacterTextSplitter 使用 len() 计数，非 token）",
    )
    chunk_overlap: int = Field(
        default=50,
        alias="RAG_CHUNK_OVERLAP",
        description="子块 overlap 字符数 (~20% of 256)",
    )
    parent_chunk_size: int = Field(
        default=1024,
        alias="RAG_PARENT_CHUNK_SIZE",
        description="父块字符数（RecursiveCharacterTextSplitter 使用 len() 计数，非 token）",
    )
    retrieval_top_k: int = Field(
        default=40,
        alias="RAG_RETRIEVAL_TOP_K",
        description="检索召回数量",
    )
    rerank_model: str = Field(
        default="qwen3-rerank",
        alias="RAG_RERANK_MODEL",
        description="DashScope Rerank 模型名 (gte-rerank 已下线，迁移至 qwen3-rerank)",
    )
    rerank_api_url: str = Field(
        default="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        alias="RAG_RERANK_API_URL",
        description="DashScope Rerank API 端点",
    )
    rerank_top_k: int = Field(
        default=8,
        alias="RAG_RERANK_TOP_K",
        description="重排序后返回数量",
    )
    rrf_k: int = Field(
        default=60,
        alias="RAG_RRF_K",
        description="RRF 平滑常数",
    )
    rrf_top_k: int = Field(
        default=20,
        alias="RAG_RRF_TOP_K",
        description="RRF 融合后返回的最大文档数",
    )

    # ── 超时配置 ─────────────────────────────────────────────
    retrieval_timeout_s: float = Field(default=5.0, alias="RAG_RETRIEVAL_TIMEOUT_S")
    rerank_timeout_s: float = Field(default=10.0, alias="RAG_RERANK_TIMEOUT_S")
    compress_timeout_s: float = Field(default=5.0, alias="RAG_COMPRESS_TIMEOUT_S")
    llm_timeout_s: float = Field(default=30.0, alias="RAG_LLM_TIMEOUT_S")
    global_timeout_s: float = Field(default=18.0, alias="RAG_GLOBAL_TIMEOUT_S")

    # ── LangFuse ─────────────────────────────────────────────
    langfuse_enabled: bool = Field(
        default=True,
        alias="RAG_LANGFUSE_ENABLED",
        description="是否启用 Langfuse 监控（默认开启；false 时所有监控代码短路跳过）",
    )
    langfuse_public_key: str = Field(
        default="pk-lf-xxx",
        alias="RAG_LANGFUSE_PUBLIC_KEY",
    )
    langfuse_secret_key: str = Field(
        default="sk-lf-xxx",
        alias="RAG_LANGFUSE_SECRET_KEY",
    )
    langfuse_host: str = Field(
        default="http://localhost:3000",
        alias="RAG_LANGFUSE_HOST",
    )

    # ── JWT 认证 ──────────────────────────────────────────────
    jwt_secret: str = Field(
        default="change-me-in-production-use-random-secret",
        alias="RAG_JWT_SECRET",
        description="JWT 签名密钥，生产环境必须修改",
    )
    jwt_expire_hours: int = Field(
        default=24,
        alias="RAG_JWT_EXPIRE_HOURS",
        description="JWT Token 过期时间（小时）",
    )
    registration_enabled: bool = Field(
        default=True,
        alias="RAG_REGISTRATION_ENABLED",
        description="是否开放用户注册",
    )
    # Deprecated: admin/user role distinction has been removed.
    # This field is kept for backward compatibility only — it no longer gates any functionality.
    admin_secret_code: str = Field(
        default="321458",
        alias="RAG_ADMIN_SECRET_CODE",
        description="[DEPRECATED] 管理员注册秘钥 — 已废弃，不再用于权限控制",
    )

    # ── 平台扩展配置 ─────────────────────────────────────────
    auth_enabled: bool = Field(
        default=True,
        alias="RAG_AUTH_ENABLED",
        description="是否启用 API Key 鉴权",
    )
    multimodal_enabled: bool = Field(
        default=False,
        alias="RAG_MULTIMODAL_ENABLED",
        description="是否启用图片/表格等多模态摄入",
    )
    max_context_tokens: int = Field(
        default=8192,
        alias="RAG_MAX_CONTEXT_TOKENS",
        description="会话历史最大上下文 token 估算上限",
    )
    summarize_after_turns: int = Field(
        default=10,
        alias="RAG_SUMMARIZE_AFTER_TURNS",
        description="会话超过多少轮后触发摘要",
    )
    session_ttl_days: int = Field(
        default=30,
        alias="RAG_SESSION_TTL_DAYS",
        description="会话保留天数，超过此天数的会话将被自动隐藏/清理",
    )

    # ── Connector 配置 ───────────────────────────────────────
    connector_slack_enabled: bool = Field(default=False, alias="RAG_CONNECTOR_SLACK_ENABLED")
    connector_slack_token: str = Field(default="", alias="RAG_CONNECTOR_SLACK_TOKEN")
    connector_slack_channels: str = Field(default="", alias="RAG_CONNECTOR_SLACK_CHANNELS")
    connector_slack_interval_minutes: int = Field(
        default=30,
        alias="RAG_CONNECTOR_SLACK_INTERVAL_MINUTES",
    )
    connector_notion_enabled: bool = Field(default=False, alias="RAG_CONNECTOR_NOTION_ENABLED")
    connector_notion_token: str = Field(default="", alias="RAG_CONNECTOR_NOTION_TOKEN")
    connector_notion_database_ids: str = Field(default="", alias="RAG_CONNECTOR_NOTION_DATABASE_IDS")
    connector_notion_interval_minutes: int = Field(
        default=60,
        alias="RAG_CONNECTOR_NOTION_INTERVAL_MINUTES",
    )

    # ── 成本估算配置（每 1K tokens 美元）──────────────────────
    model_pricing_input_per_1k: str = Field(
        default="deepseek-chat:0.00014,qwen3.6-plus:0.0003",
        alias="RAG_MODEL_PRICING_INPUT_PER_1K",
    )
    model_pricing_output_per_1k: str = Field(
        default="deepseek-chat:0.00028,qwen3.6-plus:0.0006",
        alias="RAG_MODEL_PRICING_OUTPUT_PER_1K",
    )

    # ── 速率限制 ─────────────────────────────────────────────
    rate_limit_window_s: int = Field(
        default=60,
        alias="RAG_RATE_LIMIT_WINDOW",
        description="速率限制滑动窗口大小（秒）",
    )
    rate_limit_max_requests: int = Field(
        default=30,
        alias="RAG_RATE_LIMIT_MAX",
        description="每窗口最大请求数",
    )

    # ── 文件上传 ─────────────────────────────────────────────
    max_upload_size_mb: int = Field(
        default=50,
        alias="RAG_MAX_UPLOAD_SIZE_MB",
        description="文件上传最大允许大小（MB）",
    )

    # ── Ingestion 优化 ────────────────────────────────────────
    pdf_engine: str = Field(
        default="markitdown",
        alias="RAG_PDF_ENGINE",
        description="PDF 解析引擎: markitdown (默认) 或 pypdf",
    )
    ingestion_min_quality_score: float = Field(
        default=0.30,
        alias="RAG_INGESTION_MIN_QUALITY_SCORE",
        description="文档最低质量分数（0.0-1.0），低于此值拒绝入库并返回 422",
    )
    ingestion_min_chunk_length: int = Field(
        default=20,
        alias="RAG_INGESTION_MIN_CHUNK_LENGTH",
        description="Chunk 最短字符数，低于此值的碎片在索引前静默丢弃",
    )

    # ── Ingestion: Chunk Refinement ──────────────────────────
    rag_ingestion_chunk_refinement_enabled: bool = Field(
        default=False,
        alias="RAG_INGESTION_CHUNK_REFINEMENT_ENABLED",
        description="启用 LLM Chunk 文本去噪精炼（默认关闭）",
    )
    rag_ingestion_metadata_enrichment_enabled: bool = Field(
        default=False,
        alias="RAG_INGESTION_METADATA_ENRICHMENT_ENABLED",
        description="启用 LLM Metadata 元数据丰富（默认关闭）",
    )
    rag_ingestion_llm_model: str = Field(
        default="deepseek-chat",
        alias="RAG_INGESTION_LLM_MODEL",
        description="Ingestion Refinement/Enrichment 使用的 LLM 模型名",
    )
    rag_ingestion_enrichment_concurrency: int = Field(
        default=5,
        alias="RAG_INGESTION_ENRICHMENT_CONCURRENCY",
        description="Enrichment/Refinement 并行处理的并发数",
    )
    rag_chunk_splitter: str = Field(
        default="recursive",
        alias="RAG_CHUNK_SPLITTER",
        description="Chunk 分割策略: recursive (默认) 或 semantic (预留)",
    )
    rag_bm25_index_dir: str = Field(
        default="data/db/bm25/",
        alias="RAG_BM25_INDEX_DIR",
        description="BM25 索引 JSON 文件持久化目录",
    )
    ingestion_integrity_enabled: bool = Field(
        default=True,
        alias="RAG_INGESTION_INTEGRITY_ENABLED",
        description="启用 SHA256 文件完整性校验，跳过已摄入的文件",
    )
    ingestion_deterministic_ids: bool = Field(
        default=True,
        alias="RAG_INGESTION_DETERMINISTIC_IDS",
        description="启用确定性 Chunk ID（基于文档哈希），关闭则使用 UUID",
    )
    ingestion_table_semantic: bool = Field(
        default=True,
        alias="RAG_INGESTION_TABLE_SEMANTIC",
        description="启用 Markdown 表格语义保留（原子 chunk + NL 描述）",
    )
    ingestion_max_chunks_per_file: int = Field(
        default=5000,
        alias="RAG_INGESTION_MAX_CHUNKS_PER_FILE",
        description="单文件最大 chunk 数（超出截断并告警）",
    )

    # ── 评估 (RAGAS) ──────────────────────────────────────────
    eval_default_testset_size: int = Field(
        default=10,
        alias="RAG_EVAL_DEFAULT_TESTSET_SIZE",
        description="自动生成评估测试集的默认数量",
    )

    # ── 服务 ─────────────────────────────────────────────────
    app_host: str = Field(default="0.0.0.0", alias="RAG_APP_HOST")
    app_port: int = Field(default=8000, alias="RAG_APP_PORT")
    log_level: str = Field(default="INFO", alias="RAG_LOG_LEVEL")
    @model_validator(mode="after")
    def validate_auth_secrets(self) -> "Settings":
        """Validate JWT secret and admin code strength.

        In production (ENV=production), weak secrets are always rejected
        regardless of auth_enabled. In development, weak secrets are allowed.
        """
        env = os.getenv("ENV", "development")
        if env != "production":
            return self

        if self.jwt_secret in UNSAFE_JWT_SECRETS:
            raise ValueError(
                "RAG_JWT_SECRET must be set to a strong non-placeholder value "
                "in production (ENV=production)"
            )
        return self


# 单例
settings = Settings()
