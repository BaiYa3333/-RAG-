"""MCP Server — 将企业 RAG 能力封装为 MCP 工具.

传输模式:
- stdio: ``python -m src.mcp_server``（本地 Host 子进程拉起）
- Streamable HTTP: main.py 按 RAG_MCP_ENABLED 挂载到 /mcp

实现见 server.py（FastMCP 实例与工具注册）/ service.py（传输无关业务封装）
/ auth.py（HTTP API Key 守卫）。
"""
