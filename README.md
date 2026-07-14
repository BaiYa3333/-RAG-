# 🏢 RAG_Project — 企业级知识助手

基于 **检索增强生成（RAG）** 的企业知识管理平台，支持多知识库管理、多策略混合检索、智能问答生成，帮助团队高效管理和利用企业文档知识。

## 核心能力

| 能力 | 描述 |
|------|------|
| **多格式文档摄入** | 支持 PDF、Word、Excel、PPT、EPUB、Markdown、HTML、CSV 等 10+ 格式 |
| **多知识库管理** | 按类别创建独立知识库，支持权限控制（read/write/admin） |
| **混合检索** | 稠密向量检索（ChromaDB）+ 稀疏关键词检索（自建倒排索引 BM25/jieba）+ RRF 融合 |
| **智能问答** | 基于 DeepSeek/Qwen 大模型，支持多轮对话、流式 SSE 输出 |
| **质量管控** | Tier1/Tier2 分层检索 + Quality Gate + Query Expansion (HyDE) |
| **全链路可观测** | Langfuse 集成（trace、延迟、token 用量、成本估算） |
| **Web UI** | 现代化前端界面，支持 KB 切换、文档上传、实时对话 |

## 技术栈

```
语言:       Python 3.11+
Web框架:    FastAPI + Uvicorn (ASGI)
向量数据库: ChromaDB
关系数据库: PostgreSQL 16 + pgvector
缓存:       Redis 7
工作流:     LangGraph 1.0 (StateGraph)
LLM:        DeepSeek Chat / Qwen 3.6 Plus
Embedding:  DashScope text-embedding-v4
Rerank:     DashScope qwen3-rerank
容器化:     Docker Compose (7 服务)
```

## 快速开始

### 1. 启动 Docker 服务

```bash
cd docker
docker compose up -d
```

启动后确认所有服务就绪：
```bash
docker compose ps
# 应看到: postgres, chroma, redis (以及可选的 langfuse)
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写 API Key:
#   RAG_DEEPSEEK_API_KEY=sk-xxx
#   RAG_QWEN_API_KEY=sk-xxx
#   RAG_EMBEDDING_API_KEY=sk-xxx
```

### 4. 启动应用

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 http://localhost:8000/app 即可使用 Web UI。

---

## 🧪 演示场景 — 企业知识助手一键搭建

项目提供 `scripts/seed_demo.py` 种子脚本，自动创建 **3 个分类知识库** 并填充 **~100 篇中文企业文档**，让您在 5 分钟内体验完整的企业知识助手工作流。

### 知识库结构

| 知识库 | 名称 | 文档数 | 内容范围 |
|--------|------|--------|----------|
| 📋 规章制度 | policies | ~10 | 考勤休假、薪酬绩效、信息安全、财务报销、招聘培训等 |
| 📖 产品手册 | manuals | ~45 | CloudDesk 客服、HRMS 人力、OA 协同、CRM 客户管理等 |
| 🔧 技术文档 | technical | ~45 | RAG 系统架构、前后端规范、数据库设计、运维监控等 |

文档格式分布：Markdown (~58) + PDF (~31) + TXT (~11)，覆盖多种真实企业文档形态。

### 运行种子脚本

```bash
# 完整流程：创建 KB → 生成文件 → 上传 → 验证
python scripts/seed_demo.py

# 指定并发数（默认5）
python scripts/seed_demo.py --concurrency 3

# 强制重新上传（跳过去重检测）
python scripts/seed_demo.py --force

# 分步执行
python scripts/seed_demo.py --step 1   # 仅创建知识库
python scripts/seed_demo.py --step 2   # 仅生成文件
python scripts/seed_demo.py --step 3   # 仅上传文档
python scripts/seed_demo.py --step 4   # 仅验证结果
```

脚本执行约 3-5 分钟（取决于网络和 API 限频），输出示例：

```
╔══════════════════════════════════════════════════╗
║     🏢 企业知识助手 — 演示环境一键搭建            ║
╚══════════════════════════════════════════════════╝

============================================================
  Step 1/4: 创建知识库
============================================================
  ✓ [policies] 创建成功 → "规章制度"
  ✓ [manuals] 创建成功 → "产品手册"
  ✓ [technical] 创建成功 → "技术文档"

============================================================
  Step 2/4: 生成文档文件
============================================================
  ▸ 规章制度 (10 篇)
    md/pdf       员工考勤与假期制度.md
    md/pdf       薪酬与绩效管理制度.md
    ...

============================================================
  Step 3/4: 上传文档 (并发=5)
============================================================
  [1/131] ✓ 员工考勤与假期制度.md  chunks= 18   3.2s
  [2/131] ✓ 薪酬与绩效管理制度.md  chunks= 15   2.8s
  ...

============================================================
  Step 4/4: 验证结果
============================================================
  PostgreSQL 文档统计:
  规章制度      (kb=policies   ):   14 篇文档,  185 chunks
  产品手册      (kb=manuals    ):   65 篇文档, 1150 chunks
  技术文档      (kb=technical  ):   66 篇文档, 1280 chunks

  ChromaDB 向量统计:
  kb_<id>                       :  2615 vectors

══════════════════════════════════════════════════
  演示环境搭建完成！  总耗时: 245s
══════════════════════════════════════════════════
```

### 操作流程

1. 打开浏览器访问 **http://localhost:8000/app**
2. 在左侧边栏选择要检索的知识库（可多选）
3. 输入问题，按回车发送
4. 查看 AI 回答和引用来源（点击来源可展开原文）

### 📝 演示示例问题

以下 10 个示例问题覆盖全部 3 个知识库，方便演示时直接使用：

| # | 问题 | 知识库 | 关注点 |
|---|------|--------|--------|
| 1 | 员工年假有多少天？工龄10年以上呢？ | 规章制度 | 精确数值抽取 |
| 2 | CloudDesk 专业版和企业版在功能上有什么区别？ | 产品手册 | 表格对比理解 |
| 3 | RAG 系统的混合检索是怎么工作的？ | 技术文档 | 技术概念解释 |
| 4 | 远程办公需要满足什么条件？ | 规章制度 | 条件列表提取 |
| 5 | CloudDesk 的定价方案有哪些？最低多少钱？ | 产品手册 | 表格数据提取 |
| 6 | 如果病假超过3天，工资怎么算？ | 规章制度 | 条件分支查询 |
| 7 | 私有化部署需要什么服务器配置？ | 产品手册 | 跨文档综合 |
| 8 | 数据库备份策略是什么？多久备份一次？ | 技术文档 | 具体参数提取 |
| 9 | text-embedding-v4 和 bge-large-zh 怎么选？ | 技术文档 | 对比决策 |
| 10 | 信息安全事件分级是怎么分的？严重事件多久响应？ | 规章制度 | 分级表格理解 |

---

## 项目结构

```
RAG_Project/
├── src/
│   ├── api/              # FastAPI 路由（chat, kb, document, session, auth, admin）
│   ├── rag/              # RAG 核心管线
│   │   ├── ingestion/    # 文档加载、解析、清洗、精炼
│   │   ├── indexing/     # 分块、管道编排
│   │   ├── retrieval/    # 稠密/稀疏/混合检索、RRF、重排序
│   │   ├── generation/   # LLM 生成、上下文压缩
│   │   ├── embeddings/   # text-embedding-v4 集成
│   │   └── knowledge_base/  # 知识库 CRUD 服务
│   ├── graph/            # LangGraph 工作流（意图路由、检索门控、Agent 搜索）
│   ├── llm/              # LLM 注册中心（多模型管理）
│   ├── stores/           # PostgreSQL / ChromaDB / Redis 连接管理
│   └── observability/    # Langfuse 追踪集成
├── scripts/
│   ├── seed_demo.py          # 演示种子脚本（本变更新增）
│   ├── seed_demo_content.py  # ~100 篇文档内容定义
│   ├── pdf_generator.py      # PDF 生成模块（fpdf2）
│   └── seed_knowledge_base.py   # 旧版种子脚本（默认 KB）
├── docker/               # Docker Compose + Dockerfile
├── data/
│   ├── db/               # 运行时数据
│   │   ├── chroma/       # ChromaDB 向量持久化（bind mount → 容器 /data）
│   │   ├── bm25/         # BM25 自建倒排索引 JSON 文件
│   │   └── ingestion_history.db  # 文件摄入去重记录
│   └── demo_docs/        # 演示文档生成目录
│       ├── policies/     # 规章制度文档
│       ├── manuals/      # 产品手册文档
│       └── technical/    # 技术文档文档
└── openspec/             # 项目规范与变更记录
```

## 开发

```bash
# 代码格式化
black src/ scripts/
isort src/ scripts/

# 运行测试
pytest tests/ -v

# 代码检查
ruff check src/
```
