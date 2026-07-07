"""测试 Markdown 表格语义保留 — 表格检测、NL 描述、原子 chunking."""

import pytest

from src.rag.ingestion.table_parser import TableParser, _is_separator_row, _parse_table_row


# ── is_table_block ─────────────────────────────────

def test_is_table_block_valid():
    assert TableParser.is_table_block("| A | B | C |") is True


def test_is_table_block_invalid():
    assert TableParser.is_table_block("普通文本") is False
    assert TableParser.is_table_block("inline | pipe") is False


def test_is_separator_row_valid():
    assert _is_separator_row("|---|---|") is True
    assert _is_separator_row("| :--- | ---: |") is True


# ── detect_tables ──────────────────────────────────

def test_detect_tables_single():
    text = """# 标题

| 名称 | 价格 |
|------|------|
| 产品A | 10 |
| 产品B | 20 |

后续文本。"""
    tables = TableParser.detect_tables(text)
    assert len(tables) == 1
    start, end, table_text = tables[0]
    assert "产品A" in table_text
    assert "产品B" in table_text


def test_detect_tables_multiple():
    text = """| 表1 | 值 |
|------|----|
| X | 1 |

中间文本

| 表2 | 值 |
|------|----|
| Y | 2 |"""
    tables = TableParser.detect_tables(text)
    assert len(tables) == 2


def test_detect_tables_none():
    text = """只有普通文本。
没有表格。"""
    tables = TableParser.detect_tables(text)
    assert len(tables) == 0


def test_detect_tables_ignores_inline_pipe():
    text = """管道符 | 不是表格
另一行 | 也没有分隔行"""
    tables = TableParser.detect_tables(text)
    assert len(tables) == 0


# ── generate_table_description ─────────────────────

def test_generate_table_description_simple():
    table_text = """| 产品 | 价格 | 销量 |
|------|------|------|
| A | 10 | 100 |
| B | 20 | 50 |"""
    desc = TableParser.generate_table_description(table_text)
    assert "[表格内容]" in desc
    assert "产品A价格10销量100" in desc
    assert "产品B价格20销量50" in desc
    assert desc.endswith("。")


def test_generate_table_description_empty_data():
    table_text = """| 产品 | 价格 |
|------|------|"""
    desc = TableParser.generate_table_description(table_text)
    assert desc == ""


def test_generate_table_description_wide_table():
    """超过 max_cols 列时应截断。"""
    cols = [f"col{i}" for i in range(15)]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * 15) + "|"
    row = "| " + " | ".join(["val"] * 15) + " |"
    table_text = f"{header}\n{sep}\n{row}"
    desc = TableParser.generate_table_description(table_text, max_cols=10)
    # 应该只包含前 10 列
    assert "col10" not in desc  # col10 是第 11 列（0-based: 0-9 = 10 cols）


# ── extract_table_metadata ─────────────────────────

def test_extract_table_metadata():
    table_text = """| 产品 | 价格 | 销量 |
|------|------|------|
| A | 10 | 100 |
| B | 20 | 50 |"""
    meta = TableParser.extract_table_metadata(table_text)
    assert meta["has_table"] is True
    assert meta["table_row_count"] == 2
    assert meta["table_column_count"] == 3


def test_extract_table_metadata_empty():
    table_text = "| A | B |\n|---|---|"
    meta = TableParser.extract_table_metadata(table_text)
    assert meta["has_table"] is True
    assert meta["table_row_count"] == 0


# ── _parse_table_row ───────────────────────────────

def test_parse_table_row():
    cells = _parse_table_row("| A | B | C |")
    assert cells == ["A", "B", "C"]


def test_parse_table_row_with_spaces():
    cells = _parse_table_row("|  产品A  |  10  |")
    assert cells == ["产品A", "10"]
