"""FastMCP 实例与工具注册 — rag_query / rag_search / list_knowledge_bases.

工具 docstring 面向 Host LLM 撰写（何时用我 / 何时改用兄弟工具 / 前置条件），
是 Host 侧工具选择的唯一信息来源。

异常处理：service 层抛出的 ValueError 携带可读消息（供 Host LLM 自纠正），
其余异常包装为通用可读消息 — FastMCP 将工具内异常转为 MCP tool error
(isError=true)，server 进程与其他会话不受影响。
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from src.config import settings
from src.mcp_server import service
from src.utils.logger import logger

_INSTRUCTIONS = (
    "企业 RAG 知识库服务。推荐工作流：先用 list_knowledge_bases 发现可检索的知识库，"
    "再按需选择 rag_search（取原始证据自行推理，快）或 rag_query（要完整引用答案，慢但深）。"
)


def create_mcp_server() -> FastMCP:
    """创建并注册全部工具的 FastMCP 实例。

    工厂函数而非模块级单例注册：session_manager 每实例只能 run() 一次，
    测试需要能创建独立实例。

    - streamable_http_path 保持默认 "/mcp"：main.py 以精确 Route("/mcp") 注册
      （不用 Mount 前缀挂载——Mount 对不带尾斜杠的 /mcp 会先 307 重定向，
      守卫不执行且严格客户端会失败）
    - stateless_http=True：无会话状态，兼容负载均衡与更多客户端
    - DNS rebinding 防护关闭：认证由 MCPAPIKeyGuard 承担，且服务可部署在
      任意 Host 名之后（FastMCP 默认仅放行 localhost Host 头）
    """
    mcp = FastMCP(
        name=settings.mcp_server_name,
        instructions=_INSTRUCTIONS,
        stateless_http=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @mcp.tool()
    async def rag_query(
        question: str,
        kb_ids: list[str] | None = None,
        model: str | None = None,
        language: str | None = None,
    ) -> dict:
        """对企业知识库执行完整 RAG 问答，返回带引用来源的最终答案。

        适用：需要现成的、带引用的结论；或复杂的分析/对比类问题（服务端会自动
        分解为子问题并行检索再聚合，这是 rag_search 无法提供的能力）。
        耗时可达 60 秒并消耗服务端 LLM 配额；若只需原始证据自行推理，改用 rag_search。

        Args:
            question: 自包含的完整问题。本工具无对话记忆，勿传“它怎么样”等
                依赖上下文的表述——请先在你这侧完成指代消解。
            kb_ids: 目标知识库 ID 列表。省略时检索所有可访问知识库；
                有效值请先通过 list_knowledge_bases 获取。
            model: 可选生成模型名，省略用服务端默认；传入无效值会返回可用列表。
            language: 可选答案语言（如 "zh" / "en"），省略时跟随问题语言。

        Returns:
            {answer, citations, sources: [{content, score, metadata}], intent,
            retrieval_tier}。sources 为生成所依据的检索片段。
        """
        try:
            return await service.answer_question(
                question, kb_ids=kb_ids, model=model, language=language
            )
        except ValueError:
            raise
        except Exception as exc:
            logger.warning("mcp_rag_query_error", error=str(exc))
            raise ValueError(f"rag_query 执行失败: {exc}。请稍后重试或改用 rag_search。") from exc

    @mcp.tool()
    async def rag_search(
        query: str,
        kb_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> dict:
        """在企业知识库中检索原始文档片段（混合检索+重排序），不生成答案。

        适用：你想自己阅读证据、交叉验证多个来源、或自行组织答案。秒级返回、
        不消耗服务端生成配额。每条结果约 1000 token 的完整段落上下文，含相关度
        score（越高越相关，低分结果请自行判断可用性）。空结果 (total=0) 表示
        库中无相关内容，不是错误。复杂多跳/对比类问题建议改用 rag_query。

        Args:
            query: 检索查询，语义完整的一句话效果最佳。
            kb_ids: 目标知识库 ID 列表，省略时检索所有可访问知识库；
                有效值请先通过 list_knowledge_bases 获取。
            top_k: 返回片段数（1-20，默认 5）。每条约 1000 token，请按需取量。

        Returns:
            {chunks: [{content, score, metadata: {source, kb_id, doc_id,
            chunk_id}}], total}，按相关度降序。
        """
        try:
            return await service.search_chunks(query, kb_ids=kb_ids, top_k=top_k)
        except ValueError:
            raise
        except Exception as exc:
            logger.warning("mcp_rag_search_error", error=str(exc))
            raise ValueError(f"rag_search 执行失败: {exc}。请稍后重试。") from exc

    @mcp.tool()
    async def list_knowledge_bases() -> dict:
        """列出当前可访问的知识库（id / 名称 / 描述 / 文档数）。

        在首次调用 rag_search 或 rag_query 前先调用本工具，以获取有效的
        kb_ids 并根据各库的名称与描述判断问题应检索哪些库。结果受服务身份
        权限过滤——列表之外的 kb_id 不可用。

        Returns:
            {knowledge_bases: [{id, name, description, document_count}]}
        """
        try:
            return await service.list_kbs()
        except ValueError:
            raise
        except Exception as exc:
            logger.warning("mcp_list_kbs_error", error=str(exc))
            raise ValueError(f"list_knowledge_bases 执行失败: {exc}。请稍后重试。") from exc

    return mcp


# 生产单例 — stdio 入口与 main.py HTTP 挂载共用
mcp = create_mcp_server()
