"""LLM generator - prompt assembly + streaming/non-streaming generation + fallback."""

import time
from typing import AsyncGenerator

from src.llm.factory import create_llm
from src.observability.llm_tracker import track_llm_call

SYSTEM_PROMPT = (
    "你是一个企业知识库问答助手，回答专业、清晰、有条理。\n\n"
    "回答规则：\n"
    "1. 优先参考文档内容，结合自身知识给出完整、自然的回答\n"
    "2. 引用文档中的具体数字、日期、名称时，在引用内容后标注 [^N]（N 为文档编号），如：根据产品手册，最大用户数为50人[^1]\n"
    "3. 文档信息与常识有冲突时，以文档为准并说明\n"
    "4. 对比类问题，分点列出各选项特征\n"
    "5. 如文档信息不足，用自身知识合理补充，无需刻意标注\n"
    "6. 用中文回答，保持自然流畅的表达\n"
    "7. 不要在回答末尾列出引用来源列表（系统会自动生成引用来源）\n"
    "重要：每个从文档中引用的关键事实都必须标注 [^N]，编号从1开始递增。无法溯源的内容标注「基于模型知识」。\n"
)


TABULAR_SYSTEM_PROMPT = (
    "你是一个企业知识库问答助手，擅长从文档中提取结构化数据。\n\n"
    "当问题需要对比或列举时，请按以下规则输出：\n"
    "1. 使用标准 Markdown 表格格式\n"
    "2. 表头简洁，每列含义明确\n"
    "3. 数据来源标注在表格下方：[来源: 文档名]\n"
    "4. 如果数据不完整，在对应单元格填写\"—\"\n"
    "5. 表格后可附加 1-2 句文字说明\n"
    "6. 引用文档中的具体数据时，在单元格内标注 [^N]\n"
    "7. 不要在回答末尾列出引用来源列表（系统会自动生成引用来源）\n\n"
    "示例输出格式：\n"
    "| 产品版本 | 存储空间 | 用户数上限 | 月费 |\n"
    "|---------|---------|-----------|-----|\n"
    "| 基础版   | 10GB    | 5人        | ¥99  |\n"
    "| 专业版   | 100GB   | 50人       | ¥499 |\n\n"
    "两款产品主要区别在于存储空间和用户数上限。[^1][^2]\n"
)


def _temperature_for_intent(intent: str | None) -> float:
    return {
        "factoid": 0.3,
        "comparison": 0.5,
        "summary": 0.6,
        "analytical": 0.5,
        "tabular": 0.3,
        "chitchat": 0.7,
    }.get(intent or "factoid", 0.3)


def _detect_format_mode(query: str, format_mode: str | None) -> str:
    if format_mode and format_mode != "auto":
        return format_mode
    structured_keywords = ("markdown", "md文档", "结构化", "生成文档")
    return "structured" if any(k.lower() in query.lower() for k in structured_keywords) else "lightweight"


def build_prompt(
    query: str,
    context: str,
    language: str | None = None,
    format_mode: str | None = "lightweight",
    intent: str | None = None,
) -> list[dict]:
    mode = _detect_format_mode(query, format_mode)
    language_rule = f"请使用 {language} 语言回答。" if language else "默认使用与用户问题相同的语言回答。"
    format_rule = (
        "采用自然段落，段落之间用空行分隔；除非用户要求，不使用 Markdown 标题或长列表。"
        if mode == "lightweight"
        else "采用完整 Markdown 结构，合理使用标题、列表和加粗。"
    )
    # 按意图选择 system prompt
    if intent == "tabular":
        system = TABULAR_SYSTEM_PROMPT
    else:
        system = SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"## Document Context\n\n{context}\n\n"
            f"## User Question\n\n{query}\n\n"
            "## 任务说明\n\n"
            f"请根据以上文档内容回答用户问题。如有不明确之处请标注。{language_rule}\n"
            f"格式要求：{format_rule}"
        )},
    ]


async def generate(
    query: str,
    context: str,
    model_name: str | None = None,
    language: str | None = None,
    format_mode: str = "lightweight",
    intent: str | None = None,
    **llm_kwargs,
) -> str:
    if not context.strip():
        return "No relevant documents found. Please rephrase your question or upload more relevant documents."

    llm_start = time.monotonic()
    model_id = model_name or "unknown"
    try:
        llm = create_llm(model_name=model_name)
        model_id = llm.model_id
        params = {
            **llm.default_params,
            "temperature": _temperature_for_intent(intent),
            "max_tokens": 2048,
            **llm_kwargs,
        }
        messages = build_prompt(query, context, language=language, format_mode=format_mode, intent=intent)
        resp = await llm.client.chat.completions.create(
            model=llm.model_id,
            messages=messages,
            **params,
        )
        track_llm_call(
            name="generate_answer_llm",
            model=llm.model_id,
            start_time=llm_start,
            response=resp,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        track_llm_call(
            name="generate_answer_llm",
            model=model_id,
            start_time=llm_start,
            error=str(e),
        )
        snippets = "\n---\n".join(context.split("\n")[:20])
        return (
            "Generation service temporarily unavailable. Retrieved document snippets:\n\n"
            f"{snippets}\n\n"
            f"[Error: {e}]"
        )


async def generate_stream(
    query: str,
    context: str,
    model_name: str | None = None,
    language: str | None = None,
    format_mode: str = "lightweight",
    intent: str | None = None,
    **llm_kwargs,
) -> AsyncGenerator[str, None]:
    if not context.strip():
        yield "No relevant documents found. Please rephrase your question."
        return

    llm_start = time.monotonic()
    model_id = model_name or "unknown"
    try:
        llm = create_llm(model_name=model_name)
        model_id = llm.model_id
        params = {
            **llm.default_params,
            "temperature": _temperature_for_intent(intent),
            "max_tokens": 2048,
            **llm_kwargs,
        }
        messages = build_prompt(query, context, language=language, format_mode=format_mode, intent=intent)
        stream = await llm.client.chat.completions.create(
            model=llm.model_id,
            messages=messages,
            stream=True,
            **params,
        )
        total_tokens = 0
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token_text = chunk.choices[0].delta.content
                total_tokens += len(token_text)
                yield token_text
        # Record generation span after streaming completes with approximate token count
        track_llm_call(
            name="generate_answer_llm",
            model=model_id,
            start_time=llm_start,
            metadata={"stream_mode": True, "approx_output_chars": total_tokens},
        )
        return  # stream generator finished
    except Exception as e:
        track_llm_call(
            name="generate_answer_llm",
            model=model_id,
            start_time=llm_start,
            error=str(e),
            metadata={"stream_mode": True},
        )
        snippets = "\n---\n".join(context.split("\n")[:20])
        yield (
            "\n\nGeneration service temporarily unavailable. Retrieved document snippets:\n\n"
            f"{snippets}\n\n"
            f"[Error: {e}]"
        )


def parse_citations(answer: str, documents: list[dict]) -> dict:
    """解析答案中的 [^N] 引用标记，返回结构化引用信息.

    Args:
        answer: LLM 生成的答案文本，包含 [^N] 引用标记
        documents: 检索到的文档列表

    Returns:
        {"1": {"source": "doc.pdf", "page": 3, "snippet": "...", "score": 0.85}, ...}
    """
    import re

    citations: dict[str, dict] = {}
    pattern = r'\[\^(\d+)\]'
    refs = re.findall(pattern, answer)

    for ref_num in set(refs):
        try:
            idx = int(ref_num) - 1
        except ValueError:
            continue
        if 0 <= idx < len(documents):
            doc = documents[idx]
            metadata = doc.get("metadata", {}) or {}
            citations[ref_num] = {
                "source": metadata.get("source", metadata.get("title", "")),
                "page": metadata.get("page", metadata.get("page_number", 1)),
                "snippet": (doc.get("content", "") or "")[:200],
                "score": doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0))),
            }
    return citations
