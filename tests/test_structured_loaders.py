"""测试结构化数据加载器 — CSV / Excel / JSON / PPTX / EPUB."""

import os
import pytest

from src.rag.ingestion.loader import (
    _load_csv, _load_xlsx, _load_json, _load_pptx, _load_epub,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ── CSV ────────────────────────────────────────────

def test_load_csv():
    path = os.path.join(FIXTURES_DIR, "sample.csv")
    docs = _load_csv(path)
    assert len(docs) == 10  # 10 data rows
    assert "名称: 产品A" in docs[0]["content"]
    assert docs[0]["metadata"]["source"] == "sample.csv"
    assert docs[0]["metadata"]["row"] == 1
    assert "columns" in docs[0]["metadata"]


def test_load_csv_has_all_columns():
    path = os.path.join(FIXTURES_DIR, "sample.csv")
    docs = _load_csv(path)
    expected_cols = ["名称", "价格", "销量", "类别", "日期"]
    assert docs[0]["metadata"]["columns"] == expected_cols


# ── Excel ──────────────────────────────────────────

def test_load_xlsx():
    path = os.path.join(FIXTURES_DIR, "sample.xlsx")
    docs = _load_xlsx(path)
    assert len(docs) > 0
    # 应包含两个 sheet 的数据
    sheets = {d["metadata"]["sheet_name"] for d in docs}
    assert "销售数据" in sheets
    assert "库存数据" in sheets


def test_load_xlsx_sheet_metadata():
    path = os.path.join(FIXTURES_DIR, "sample.xlsx")
    docs = _load_xlsx(path)
    sheet1_docs = [d for d in docs if d["metadata"]["sheet_name"] == "销售数据"]
    assert len(sheet1_docs) == 4  # 4 data rows
    assert sheet1_docs[0]["metadata"]["sheet_index"] == 0


# ── JSON ───────────────────────────────────────────

def test_load_json():
    path = os.path.join(FIXTURES_DIR, "sample.json")
    docs = _load_json(path)
    assert len(docs) == 5  # 5 objects
    assert "id: 1" in docs[0]["content"]
    assert docs[0]["metadata"]["source"] == "sample.json"
    assert docs[0]["metadata"]["index"] == 0


def test_load_json_nested_flattening():
    path = os.path.join(FIXTURES_DIR, "sample.json")
    docs = _load_json(path)
    # 嵌套对象应被展平，带有 dot-notation
    content_0 = docs[0]["content"]
    assert "address.city: 北京" in content_0 or "city: 北京" in content_0


# ── PPTX ───────────────────────────────────────────

def test_load_pptx():
    path = os.path.join(FIXTURES_DIR, "sample.pptx")
    docs = _load_pptx(path)
    assert len(docs) == 3  # 3 slides
    assert docs[0]["metadata"]["slide"] == 1
    assert docs[1]["metadata"]["slide"] == 2
    assert docs[2]["metadata"]["slide"] == 3


def test_load_pptx_title_extraction():
    path = os.path.join(FIXTURES_DIR, "sample.pptx")
    docs = _load_pptx(path)
    # 第一张 slide 标题为 "项目介绍"
    assert "项目介绍" in docs[0]["content"]


# ── EPUB ───────────────────────────────────────────

def test_load_epub():
    path = os.path.join(FIXTURES_DIR, "sample.epub")
    docs = _load_epub(path)
    assert len(docs) > 0
    # 应包含章节内容
    all_text = " ".join(d["content"] for d in docs)
    assert "第一章" in all_text or "引言" in all_text


def test_load_epub_chapter_metadata():
    path = os.path.join(FIXTURES_DIR, "sample.epub")
    docs = _load_epub(path)
    sources = {d["metadata"]["source"] for d in docs}
    assert "sample.epub" in sources
