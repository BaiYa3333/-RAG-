# ── RAG_Project Dockerfile ──────────────────────────────────
# 多阶段构建: builder 安装依赖 → runtime 运行应用
# 启动时自动运行 alembic migration

FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

# 配置 pip 国内镜像加速（阿里云）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com

# 先装 torch CPU 版，避免 sentence-transformers 拉入 CUDA（~3GB nvidia 驱动）
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu --default-timeout=120

# 装其余依赖；sentence-transformers 检测到 torch 已安装，跳过 GPU 版
RUN pip install --no-cache-dir --user -r requirements.txt --default-timeout=120

# 清理冗余文件
RUN find /root/.local -name "*.pyc" -delete && \
    find /root/.local -name "__pycache__" -type d -exec rm -rf {} + && \
    find /root/.local -name "tests" -type d -exec rm -rf {} + 2>/dev/null; true


FROM python:3.11-slim-bookworm

# 安装 postgresql-client（entrypoint 等待 DB 就绪需要 psql）
# 跳过 apt-get update: Debian CDN 在本环境不可达，直接用 base image 内置的包索引
RUN apt-get install -y --no-install-recommends --allow-unauthenticated postgresql-client 2>&1 || \
    (echo "WARNING: postgresql-client install failed, entrypoint will use pg_isready fallback" && true) && \
    rm -rf /var/lib/apt/lists/*

# 安装中文字体（PDF 种子脚本 + 文档渲染需要）
RUN apt-get install -y --no-install-recommends --allow-unauthenticated fonts-noto-cjk 2>&1 || \
    (echo "WARNING: fonts-noto-cjk install failed, PDF generation will use fpdf2 built-in font" && true) && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local

# 配置 pip 国内镜像加速并安装运行时额外依赖
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com && \
    pip install --no-cache-dir psycopg2-binary --default-timeout=120

# 预装 spaCy 英文模型 — en_core_web_sm 是 unstructured 文本分析的刚需
RUN pip install --no-cache-dir en-core-web-sm==3.8.0 -f https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl --default-timeout=120

COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY docker/entrypoint.sh ./entrypoint.sh

# 清理从主机复制过来的 __pycache__（避免跨 Python 版本的 .pyc 干扰）
RUN find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true

RUN chmod +x ./entrypoint.sh && chown -R appuser:appuser /app

USER appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV POSTGRES_HOST=postgres

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
