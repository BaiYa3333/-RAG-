"""查询拓展 — HyDE 假设文档生成."""

import time

from src.observability.langfuse_context import langfuse_context

from src.llm.factory import create_llm
from src.observability.decorators import trace_rag_node
from src.observability.llm_tracker import track_llm_call

HYDE_PROMPT = (
    "Given the following question, write a short passage that answers it. "
    "Keep it factual and concise.\n\nQuestion: {query}\n\nPassage:"
)


@trace_rag_node(name="hyde_query_expansion")
async def expand_query(query: str, model_name: str | None = None) -> str:
    llm = create_llm(model_name=model_name)  # 不传 temperature/max_tokens，避免与 create() 参数冲突
    llm_start = time.monotonic()
    resp = await llm.client.chat.completions.create(
        model=llm.model_id,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
        max_tokens=200,
        temperature=0.0,
    )
    expanded = resp.choices[0].message.content.strip()
    track_llm_call(
        name="hyde_expansion_llm",
        model=llm.model_id,
        start_time=llm_start,
        response=resp,
    )
    langfuse_context.update_current_observation(
        retrieval_type="hyde",
        original_query_len=len(query),
        expanded_query_len=len(expanded),
    )
    return expanded
