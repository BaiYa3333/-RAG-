# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Docker services (from docker/ directory)
cd docker && docker compose up -d              # start all 6 services
docker compose ps                               # check health status
docker compose up -d --build                    # rebuild + restart all
docker compose down                             # stop all

# App (local dev, after Docker services are up)
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Testing
pytest tests/ -v                                # all tests
pytest tests/ -v -k "test_name_pattern"         # single test
pytest tests/ -v --cov=src --cov-report=html    # with coverage

# Lint & format
ruff check src/                                 # lint
black src/ scripts/                             # format
isort src/ scripts/                             # import sort

# Database
cd docker && alembic upgrade head               # run migrations (via entrypoint normally)
alembic revision --autogenerate -m "message"    # create new migration

# Demo seeding
python scripts/seed_demo.py                     # full demo (3 KBs, ~100 docs)
python scripts/seed_demo.py --step 3 --force    # re-upload only
```

## Architecture

This is an enterprise RAG system built on **FastAPI + LangGraph** with multi-tier hybrid retrieval.

### Request Flow (QA Pipeline)

```
User Query → Query Condenser (coreference resolution + memory injection)
           → Intent Router (LLM: factoid/analytical/comparison/summary/tabular/chitchat)
              │
              ├─ analytical  → Agent (decompose → parallel search → aggregate)
              ├─ chitchat    → Direct LLM answer → END
              └─ others      → Tier1 Retrieval (dense + keyword boost)
                                   │
                              Quality Gate (score threshold check)
                                   │
                         ┌─────────┼──────────┐
                         │ pass    │ escalate │ fallback_generate
                         ▼         ▼          │
                    [Rerank]   Tier2          │
                      │       (HyDE+dense     │
                      │        +BM25+RRF)     │
                      │          │            │
                      │     Quality Gate ─────┘
                      │          │
                      ▼          ▼
                   [Compress] → Generate → END
```

### LangGraph StateGraph (`src/graph/`)

10 nodes wired in `workflow.py` with 4 conditional edges. State is a `TypedDict` (15+ fields) with `operator.add` reducers for accumulative fields (documents, gate_log, errors). All nodes wrapped in `safe_node()` with per-node timeouts and graceful degradation. Checkpoints via `InMemorySaver`. Global workflow timeout: 60s.

### Retrieval Pipeline (`src/rag/retrieval/`)

**Tier 1**: Dense vector search (ChromaDB cosine) + keyword boost  
**Tier 2**: HyDE query expansion → parallel dense + BM25 sparse → RRF fusion (k=60)  
**Reranker**: DashScope qwen3-rerank API (with LLM listwise → keyword identity fallback chain)

BM25 is a **self-built inverted index** (not `rank_bm25`), using jieba tokenization + RSJ IDF, persisted atomically as JSON to `data/db/bm25/`.

### Ingestion Pipeline (`src/rag/ingestion/` + `src/rag/indexing/`)

7-phase pipeline orchestrated by `IndexingPipeline.run()`:
```
Load (multi-engine) → Parse+Clean (dedup, injection detection) → Quality Check (0.30 threshold)
→ Chunk (parent 1024/child 256, deterministic IDs) → [Refine (LLM denoise)] → [Enrich (LLM title/summary/tags)]
→ Embed (DashScope API + Redis cache) → Store (ChromaDB + PostgreSQL)
```

File integrity tracked via SQLite at `data/db/ingestion_history.db`.

### Storage Layer (`src/stores/`)

| Store | Tech | Access Pattern |
|-------|------|---------------|
| Vector | ChromaDB (HTTP service) | `AsyncHttpClient`, cosine space, per-KB/per-session collections |
| Document | PostgreSQL 16 + pgvector | `asyncpg` pool, UUID PK, JSONB metadata |
| Cache | Redis 7 | Embedding cache (SHA256, TTL 1h), rate-limit counters |

ChromaDB runs as a **separate HTTP service** (not embedded). Docker bind mount: `data/db/chroma → /data`.

### Configuration (`src/config.py`)

Single `Settings` class via `pydantic-settings`, all env vars prefixed `RAG_`. 16 logical groups: LLM, Embedding, ChromaDB, PostgreSQL, Redis, Retrieval, Reranker, Quality Gate, Chunking, Timeouts, Langfuse, JWT Auth, Ingestion, Rate Limit, Model Pricing, Connectors.

### Observability (`src/observability/`)

**Langfuse v2 SDK** (must stay v2.x — v3 server uses OTLP which the v2 server doesn't support). Integration points:
- `@trace_rag_node` decorator on every graph node + API routes
- `track_llm_call()` after every LLM API call
- `langfuse_context.update_current_observation()` for node-level metadata

### Knowledge Base & Permissions

KB CRUD in PostgreSQL (`knowledge_bases` + `kb_permissions` tables). Three roles: `read`/`write`/`admin`. Each KB maps to a ChromaDB collection `rag_kb_{id}`. Collections named via `kb_collection_name()` with SHA256 truncation to 63-char limit.

## Key Constraints

- **Langfuse SDK pinned to v2.x** (`langfuse>=2.50.0,<3.0.0`) — v4 uses OTLP, incompatible with langfuse/langfuse:2 server
- **ChromaDB is HTTP service**, not embedded library — always use `AsyncHttpClient(host, port)`, never `PersistentClient`
- **BM25 is self-built** inverted index — do not import `rank_bm25`; the library is in requirements but the code uses the custom implementation in `sparse.py`
- **Deterministic chunk IDs** enabled by default (`RAG_INGESTION_DETERMINISTIC_IDS=true`) — format: `{doc_hash[:12]}_p{pi}_c{ci}_{content_hash[:8]}`
- **JWT auth** required in production; `registration_enabled` defaults false; weak secrets rejected at startup
- **Rate limiting** on `/chat` and `/chat/stream` only — Redis sliding window, fail-open if Redis unavailable
- **Python 3.11+** required
- **Mirrors**: pip configured for aliyun mirrors in Docker builds (China-optimized)

## Test Patterns

Tests use `pytest-asyncio` in `auto` mode. Key fixtures in `conftest.py`:
- `mock_llm_client` — `MagicMock` returning preset chat responses
- `mock_embedding_client` — `AsyncMock` returning fixed 1024-dim vectors
- `mock_graph` — `AsyncMock` returning preset `RAGState` with `ainvoke` + `astream`
- `test_app` — FastAPI `TestClient` with mocked state injected, auth disabled

Create new test modules in `tests/` root or subdirectories (`tests/test_api/`, `tests/test_evaluation/`).
