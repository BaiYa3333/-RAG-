"""文本清洗 — 去噪、去重、注入检测."""

import re

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|foregoing)\s+(instructions?|directives?|commands?)",
    r"system\s*prompt",
    r"you\s+are\s+now\s+a",
    r"forget\s+everything",
    r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|constraints?)",
]


def clean_text(text: str) -> str:
    text = _remove_repeated_newlines(text)
    text = _remove_control_chars(text)
    return text.strip()


def detect_injection(text: str) -> list[str]:
    flags = []
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            flags.append(pattern)
    return flags


def deduplicate_sections(sections: list[dict], threshold: float = 0.85) -> list[dict]:
    from difflib import SequenceMatcher

    seen = []
    seen_hashes: set[int] = set()  # 精确重复快速过滤

    for section in sections:
        content = section.get("content", "")

        # 精确重复：hash 秒杀
        h = hash(content)
        if h in seen_hashes:
            continue

        # 近似重复：只与最近 50 条比较（滑动窗口），避免 O(n²)
        window = seen[-50:]
        is_dup = False
        for existing in window:
            ratio = SequenceMatcher(None, content, existing.get("content", "")).ratio()
            if ratio >= threshold:
                is_dup = True
                break

        if not is_dup:
            seen_hashes.add(h)
            seen.append(section)

    return seen


def compute_quality_score(text: str) -> float:
    """计算文本质量分数 (0.0-1.0)，基于有意义字符占比。

    "有意义字符" 包括:
    - 可打印 ASCII (字母、数字、标点、空格) — U+0020..U+007E
    - CJK 统一汉字 — U+4E00..U+9FFF
    - CJK 兼容汉字 — U+F900..U+FAFF
    - 全角标点 / 假名 — U+3000..U+303F, U+FF00..U+FFEF
    - 格式化空白 (Tab / LF / CR) — 这些是合法文档结构

    垃圾指示器:
    - U+FFFD 替换字符 — 显式计为 0 分
    - 控制字符 / 代理对 / 私有区 — 不匹配以上任何范围，隐式为非有意义

    返回 [0.0, 1.0] 区间的浮点数，越高越好。
    """
    if not text:
        return 0.0

    total = len(text)
    if total == 0:
        return 0.0

    meaningful = 0

    for ch in text:
        cp = ord(ch)
        # Printable ASCII
        if 0x20 <= cp <= 0x7E:
            meaningful += 1
        # CJK Unified Ideographs
        elif 0x4E00 <= cp <= 0x9FFF:
            meaningful += 1
        # CJK Compatibility Ideographs
        elif 0xF900 <= cp <= 0xFAFF:
            meaningful += 1
        # Full-width punctuation / kana / symbols
        elif (0x3000 <= cp <= 0x303F) or (0xFF00 <= cp <= 0xFFEF):
            meaningful += 1
        # Whitespace: tab, LF, CR
        elif cp in (0x09, 0x0A, 0x0D):
            meaningful += 1
        # U+FFFD replacement character — explicitly zero
        elif cp == 0xFFFD:
            meaningful += 0
        # Everything else: non-meaningful

    ratio = meaningful / total
    return round(ratio, 4)


def filter_short_chunks(chunks: list[dict], min_length: int) -> list[dict]:
    """过滤 content 长度小于 min_length 的 chunk。

    返回新列表，不修改输入。min_length <= 0 时原样返回。
    """
    if min_length <= 0:
        return chunks
    return [c for c in chunks if len(c.get("content", "") or "") >= min_length]


def _remove_repeated_newlines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


def _remove_control_chars(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
