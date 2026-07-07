"""结构化文档解析器."""


def parse_document(elements: list[dict]) -> list[dict]:
    parsed = []
    for el in elements:
        content = el["content"]
        metadata = dict(el.get("metadata", {}))

        content = _normalize_text(content)
        metadata["has_table"] = bool(_detect_table(content))

        parsed.append({"content": content, "metadata": metadata})

    return parsed


def _normalize_text(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned.append(stripped)
    return "\n".join(cleaned)


def _detect_table(text: str) -> bool:
    return "|" in text and "\n|" in text
