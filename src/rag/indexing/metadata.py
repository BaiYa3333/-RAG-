"""元数据自动提取."""

from pathlib import Path


def extract_metadata(file_path: str, doc_elements: list[dict] | None = None) -> dict:
    path = Path(file_path)
    meta = {
        "source": path.name,
        "doc_type": path.suffix.lower().lstrip("."),
        "title": path.stem,
    }

    if doc_elements:
        pages = {el.get("metadata", {}).get("page", 1) for el in doc_elements if el.get("metadata")}
        meta["page_count"] = len(pages)

        for el in doc_elements:
            section = el.get("metadata", {}).get("section", "")
            if section:
                sections = meta.get("_sections", [])
                if section not in sections:
                    sections.append(section)
                meta["sections"] = ", ".join(sections)

    return meta
