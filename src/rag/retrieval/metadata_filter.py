"""元数据过滤条件构造."""


def build_metadata_filter(source: str | None = None, doc_type: str | None = None,
                          date_range: tuple[str, str] | None = None,
                          **extra) -> dict | None:
    conditions = []

    if source:
        conditions.append({"source": source})
    if doc_type:
        conditions.append({"doc_type": doc_type})
    for key, value in extra.items():
        if value is not None:
            conditions.append({key: value})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
