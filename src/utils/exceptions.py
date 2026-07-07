"""RAG_Project 异常层次体系。

所有自定义异常继承自 RAGException，
FastAPI exception handler 按类型映射 HTTP 状态码，
LangGraph fallback 按类型决定降级策略。
"""


class RAGException(Exception):
    """RAG 系统基础异常。"""
    ...


class IngestionError(RAGException):
    """文档摄取阶段错误 — 解析失败、格式不支持、清洗异常。"""
    ...


class IngestionQualityError(IngestionError):
    """文档质量不达标 — 乱码/无效字符过多，拒绝入库。"""
    ...


class RetrievalError(RAGException):
    """检索阶段错误 — 向量库不可用、索引未就绪、查询格式异常。"""
    ...


class GenerationError(RAGException):
    """生成阶段错误 — LLM 调用失败、prompt 格式异常、输出校验失败。"""
    ...


class StoreError(RAGException):
    """存储层错误 — 连接失败、写入异常、表不存在。"""
    ...


class LLMError(RAGException):
    """LLM 服务错误 — API 超时、限流、认证失败、模型不可用。"""
    ...
