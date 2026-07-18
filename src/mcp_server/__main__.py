"""stdio 传输入口 — ``python -m src.mcp_server``.

供本地 MCP Host（Claude Desktop / Claude Code / Cursor 等）以子进程拉起。
独立于 FastAPI 进程运行：检索模块的懒加载连接（ChromaDB / embedding cache /
BM25 索引）自行建立，KB 服务按需自建 DocStore 连接。

stdout 是 JSON-RPC 通道 — 两条硬约束：
1. 日志必须走 stderr（structlog 默认 PrintLogger 写 stdout，此处显式重定向）
2. 输出强制 UTF-8（Windows GBK 控制台会破坏 JSON-RPC 帧；Host 配置中
   建议同时设置 PYTHONUTF8=1）
"""

import sys


def main() -> None:
    # Windows GBK 防护 — stdio JSON-RPC 帧必须是 UTF-8
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

    from src.config import settings
    from src.utils.logger import setup_logger

    # 日志重定向到 stderr，避免污染 stdout 的 JSON-RPC 通道
    setup_logger(settings.log_level, stream=sys.stderr)

    from src.mcp_server.server import mcp
    from src.utils.logger import logger

    logger.info("mcp_stdio_starting", server_name=settings.mcp_server_name)
    try:
        mcp.run(transport="stdio")
    finally:
        # 退出前冲刷 Langfuse trace 数据
        try:
            from src.observability.client import flush_langfuse

            flush_langfuse()
        except Exception:
            pass
        logger.info("mcp_stdio_stopped")


if __name__ == "__main__":
    main()
