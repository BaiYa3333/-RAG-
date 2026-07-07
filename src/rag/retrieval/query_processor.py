"""查询处理 — 从自然语言查询中解析 key:value 过滤语法.

支持从查询字符串中提取过滤条件并生成 ChromaDB where 子句。
解析后的过滤条件从原始查询中剥离，剩余文本作为纯语义查询。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── 过滤键映射 ──────────────────────────────────────────
# 将用户友好的短键名映射到标准内部键名
KEY_ALIASES: dict[str, str] = {
    "collection": "collection",
    "col": "collection",
    "c": "collection",
    "type": "doc_type",
    "doc_type": "doc_type",
    "t": "doc_type",
    "tag": "tags",
    "tags": "tags",
    "source": "source",
    "src": "source",
    "s": "source",
}

# 正则: 匹配 key:value（key 为单词字符，value 为非空白字符序列或带引号的字符串）
_FILTER_RE = re.compile(
    r'(\w+):(?:("[^"]*")|(\'[^\']*\')|([^\s]+))'
)


@dataclass
class ParsedQuery:
    """解析后的查询结果."""
    clean_query: str
    """剥离过滤条件后的纯语义查询字符串."""
    filters: dict[str, str | list[str]] = field(default_factory=dict)
    """解析出的过滤条件，键为标准化内部键名."""


class QueryProcessor:
    """查询过滤器解析器.

    从自然语言查询中提取 key:value 过滤条件：
    - collection / col / c → ChromaDB collection 名过滤
    - type / doc_type / t → 文档类型过滤
    - tag / tags → 标签过滤（逗号分隔的多值）
    - source / src / s → 来源文件过滤
    - 未识别的键 → 作为通用 metadata 过滤条件

    Usage:
        processor = QueryProcessor()
        parsed = processor.parse("collection:技术文档 type:pdf tag:架构 微服务设计")
        # parsed.clean_query = "微服务设计"
        # parsed.filters = {"collection": "技术文档", "doc_type": "pdf", "tags": ["架构"]}
    """

    @staticmethod
    def parse(query: str) -> ParsedQuery:
        """解析查询字符串，提取过滤条件并剥离.

        Args:
            query: 原始用户查询字符串

        Returns:
            ParsedQuery 对象，包含 clean_query 和 filters
        """
        if not query or not query.strip():
            return ParsedQuery(clean_query="", filters={})

        filters: dict[str, str | list[str]] = {}
        matched_spans: list[tuple[int, int]] = []

        for match in _FILTER_RE.finditer(query):
            key = match.group(1).lower()
            # 提取 value（可能是带引号或不带引号的形式）
            value = match.group(2) or match.group(3) or match.group(4)
            if value:
                # 去除引号
                value = value.strip('"').strip("'")

            if not value:
                continue

            canonical_key = KEY_ALIASES.get(key, key)  # 未识别的键保持原样

            # Tags 支持逗号分隔的多值
            if canonical_key == "tags":
                existing = filters.get("tags", [])
                if isinstance(existing, str):
                    existing = [existing]
                tag_values = [t.strip() for t in value.split(",") if t.strip()]
                filters["tags"] = existing + tag_values
            else:
                filters[canonical_key] = value

            matched_spans.append((match.start(), match.end()))

        # 从原始查询中剥离已匹配的过滤片段
        clean_query = _strip_spans(query, matched_spans)

        return ParsedQuery(clean_query=clean_query, filters=filters)

    @staticmethod
    def to_chromadb_where(filters: dict[str, str | list[str]]) -> dict | None:
        """将解析出的 filters 转换为 ChromaDB where 子句.

        注意: collection 过滤不通过 where 子句，而是在检索时选择不同的 collection。
        此方法仅处理 metadata 级别的过滤（doc_type, tags, source, 及其他通用键）。

        Args:
            filters: parse() 返回的 filters 字典

        Returns:
            ChromaDB where 子句字典，或 None 如果没有可用的 metadata 过滤条件
        """
        if not filters:
            return None

        conditions: list[dict] = []

        for key, value in filters.items():
            # collection 不在 metadata where 中处理
            if key == "collection":
                continue

            if isinstance(value, list):
                # 多值过滤：使用 $and + $contains 组合
                # ChromaDB 的 $contains 需要 exact match，分别检查
                for v in value:
                    conditions.append({key: {"$eq": v}})
            else:
                conditions.append({key: {"$eq": value}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def get_collection_name(filters: dict[str, str | list[str]]) -> str | None:
        """从 filters 中提取 collection 过滤值（若有）.

        Args:
            filters: parse() 返回的 filters 字典

        Returns:
            collection 名称，或 None
        """
        col = filters.get("collection")
        if isinstance(col, list):
            return col[0] if col else None
        return col if isinstance(col, str) else None


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """从文本中删除指定的区间，并清理多余空格.

    Args:
        text: 原始文本
        spans: 要删除的 (start, end) 区间列表（已排序）

    Returns:
        清理后的文本
    """
    if not spans:
        return text.strip()

    # 按起始位置排序
    spans = sorted(spans, key=lambda s: s[0])

    result_parts: list[str] = []
    pos = 0

    for start, end in spans:
        if start > pos:
            result_parts.append(text[pos:start])
        pos = end

    # 追加剩余文本
    if pos < len(text):
        result_parts.append(text[pos:])

    result = "".join(result_parts)
    # 清理多余空格
    result = re.sub(r'\s+', ' ', result)
    return result.strip()
