"""Markdown 表格语义保留 — 表格检测、原子边界、NL 描述生成."""

import re

from src.config import settings


class TableParser:
    """Markdown 表格解析器 — 检测表格块、生成自然语言描述、提取元数据。"""

    @staticmethod
    def is_table_block(line: str) -> bool:
        """判断单行是否为 Markdown 表格行（包含 | 分隔符）。"""
        return bool(re.match(r"^\s*\|.+\|\s*$", line))

    @staticmethod
    def detect_tables(text: str) -> list[tuple[int, int, str]]:
        """检测文档中所有 Markdown 表格块。

        Returns:
            list of (start_line_idx, end_line_idx, table_text) tuples。
            start_line_idx / end_line_idx 为 0-based 行号（含）。
        """
        if not settings.ingestion_table_semantic:
            return []

        lines = text.split("\n")
        tables = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if not TableParser.is_table_block(line):
                i += 1
                continue

            # 检查下一行是否为分隔行 |---|---|
            if i + 1 >= len(lines) or not _is_separator_row(lines[i + 1]):
                i += 1
                continue

            # 找到表格起始行（包含前置标题行）
            start = i
            # 如果前一行是非空非表格行，可能是表格标题/说明
            if i > 0 and lines[i - 1].strip() and not TableParser.is_table_block(lines[i - 1]):
                start = i - 1

            # 找到表格结束行
            end = i + 1
            while end + 1 < len(lines) and TableParser.is_table_block(lines[end + 1]):
                end += 1

            table_text = "\n".join(lines[start:end + 1])
            tables.append((start, end, table_text))
            i = end + 1

        return tables

    @staticmethod
    def generate_table_description(table_text: str, max_cols: int = 10) -> str:
        """将 Markdown 表格转换为自然语言描述。

        示例输入:
        | 产品 | 价格 | 销量 |
        |------|------|------|
        | A    | 10   | 100  |
        | B    | 20   | 50   |

        示例输出:
        "[表格内容]: 产品A价格10元销量100件；产品B价格20元销量50件。"
        """
        lines = table_text.strip().split("\n")
        if len(lines) < 2:
            return ""

        # 解析行
        parsed_rows = []
        for line in lines:
            if _is_separator_row(line):
                continue
            cells = _parse_table_row(line)
            if cells:
                parsed_rows.append(cells)

        if len(parsed_rows) < 2:
            return ""  # 仅有表头无数据

        headers = parsed_rows[0]
        data_rows = parsed_rows[1:]
        # 截断列数
        headers = headers[:max_cols]

        descriptions = []
        for row in data_rows:
            row_cells = row[:max_cols]
            # 补齐不足列
            while len(row_cells) < len(headers):
                row_cells.append("")
            parts = []
            for h, v in zip(headers, row_cells):
                parts.append(f"{h}{v}")
            descriptions.append("".join(parts))

        if not descriptions:
            return ""

        return "[表格内容]: " + "；".join(descriptions) + "。"

    @staticmethod
    def extract_table_metadata(table_text: str) -> dict:
        """从 Markdown 表格文本提取元数据。"""
        lines = table_text.strip().split("\n")
        data_rows = 0
        col_count = 0

        for line in lines:
            if _is_separator_row(line):
                cells = _parse_table_row(line)
                col_count = len(cells)
            elif TableParser.is_table_block(line):
                cells = _parse_table_row(line)
                if col_count == 0:
                    col_count = len(cells)
                data_rows += 1

        # 数据行不含表头
        data_rows = max(0, data_rows - 1)

        return {
            "has_table": True,
            "table_row_count": data_rows,
            "table_column_count": col_count,
        }


def _is_separator_row(line: str) -> bool:
    """判断是否为 Markdown 表格分隔行: |---|---|"""
    return bool(re.match(r"^\s*\|[\s\-:]+\|", line))


def _parse_table_row(line: str) -> list[str]:
    """解析单行表格为单元格列表。"""
    # 去除首尾 | 和空白
    stripped = line.strip().strip("|")
    cells = [c.strip() for c in stripped.split("|")]
    return cells
