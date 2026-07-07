"""上下文压缩 — LLM 提取查询相关句子."""

import time

from src.llm.factory import create_llm
from src.config import settings
from src.observability.llm_tracker import track_llm_call

COMPRESS_PROMPT = (
    "Given the following context and question, extract only the sentences "
    "that are relevant to answering the question. Return the extracted sentences "
    "as-is, preserving source markers if present.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Relevant sentences:"
)

# 压缩器最大输入 token 估算 (留出 prompt 开销 + 输出 budget)
# 保守取 settings.max_context_tokens 的 60%，避免 LLM API 截断或报错
COMPRESS_MAX_INPUT_CHARS = int((settings.max_context_tokens or 8192) * 0.6 * 2)


def _estimate_chars_per_doc(documents: list[dict], max_total: int) -> list[int]:
    """按文档数量均分最大字符配额，每个文档不超过其原始长度。"""
    per_doc = max_total // max(len(documents), 1)
    limits = []
    for d in documents:
        content_len = len(d.get("content", ""))
        limits.append(min(content_len, per_doc))
    return limits


async def compress(
    query: str,
    documents: list[dict],
    fail_silently: bool = True,
    model_name: str | None = None,
) -> str:
    if not documents:
        return ""

    # 上下文长度保护：估算截断避免超过模型输入限制
    char_limits = _estimate_chars_per_doc(documents, COMPRESS_MAX_INPUT_CHARS)
    context_parts = []
    for i, d in enumerate(documents):
        content = d.get("content", "")
        limit = char_limits[i]
        truncated = content[:limit] if len(content) > limit else content
        context_parts.append(f"[{i+1}] {truncated}")
    context = "\n\n---\n\n".join(context_parts)

    llm_start = time.monotonic()
    model_id = model_name or settings.default_llm_model
    try:
        llm = create_llm(model_name=model_name)
        model_id = llm.model_id
        params = {
            **llm.default_params,
            "temperature": 0.0,
            "max_tokens": 1024,
        }
        resp = await llm.client.chat.completions.create(
            model=llm.model_id,
            messages=[{"role": "user", "content": COMPRESS_PROMPT.format(
                context=context, query=query)}],
            **params,
        )
        track_llm_call(
            name="compress_context_llm",
            model=llm.model_id,
            start_time=llm_start,
            response=resp,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        if fail_silently:
            track_llm_call(
                name="compress_context_llm",
                model=model_id,
                start_time=llm_start,
                error="compress_llm_failed",
            )
            return context
        raise
