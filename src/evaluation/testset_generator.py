"""RAGAS 测试集自动生成 — 从知识库文档生成 question + ground_truth 对."""

import json
from dataclasses import dataclass, field

from src.utils.logger import logger


@dataclass
class TestsetItem:
    """单条测试用例."""
    question: str
    ground_truth: str
    reference_contexts: list[str] = field(default_factory=list)


_GENERIC_QUESTION_PHRASES = (
    "这段知识库片段",
    "该知识库片段",
    "根据知识库片段",
    "这段内容",
    "上述内容",
    "该文档",
    "本文档",
)


def _is_valid_generated_case(question: str, ground_truth: str,
                             reference_contexts: list[str]) -> bool:
    """判断自动生成的 RAGAS 用例是否可回答、可检索。"""
    q = (question or "").strip()
    gt = (ground_truth or "").strip()
    refs = [r for r in reference_contexts if r and r.strip()]

    if len(q) < 8 or len(gt) < 20 or not refs:
        return False

    if any(phrase in q for phrase in _GENERIC_QUESTION_PHRASES):
        return False

    # “以下哪一项”类问题必须带选项，否则 RAG 无法知道候选项。
    if "以下哪一项" in q and not any(mark in q for mark in ("A.", "A、", "选项", "：")):
        return False

    return True


async def generate_testset(
    doc_store,
    vector_store,
    llm_client,
    document_ids: list[str] | None = None,
    size: int = 10,
) -> list[TestsetItem]:
    """从知识库文档自动生成 RAGAS 测试集。

    流程：
    1. 从 PG doc_store 拉取文档元数据 + 内容摘要
    2. 从 ChromaDB 拉取代表性 chunk 作为 reference_contexts
    3. 调用 LLM 为每份文档生成 question + ground_truth 对

    Args:
        doc_store: DocStore 实例
        vector_store: VectorStore 实例
        llm_client: LLM client (create_llm 返回)
        document_ids: 指定文档 ID 列表，None 表示全部
        size: 生成测试用例数量

    Returns:
        list[TestsetItem]
    """
    from openai import AsyncOpenAI
    from src.llm.factory import LLMClient

    try:
        # 1. 从 PG 获取文档列表
        if document_ids:
            docs = []
            for did in document_ids:
                doc = await doc_store.get_document(did)
                if doc:
                    docs.append(doc)
        else:
            # 获取最近 size 份文档
            query = "SELECT id, title, source, doc_type, chunk_count, kb_id, metadata FROM documents ORDER BY created_at DESC LIMIT $1"
            docs = await doc_store.fetch(query, size)

        if not docs:
            logger.warning("testset_generate_no_docs", document_ids=document_ids)
            return []

        logger.info("testset_gen_docs_fetched", count=len(docs))

        # 限制生成数量
        docs = docs[:size]

        # 2. 为每份文档获取代表性 chunk 作为 reference
        from src.rag.retrieval.collections import kb_collection_name
        testset: list[TestsetItem] = []
        for doc in docs:
            try:
                # 从 ChromaDB 检索文档的代表性 chunks
                # 使用 kb_id 对应的 collection，而非默认的 rag_docs_dev
                kb_id = doc.get("kb_id")
                col_name = kb_collection_name(str(kb_id)) if kb_id else "rag_docs_dev"
                chunks = await vector_store.get_by_metadata(
                    metadata_filter={"source": doc["source"]},
                    top_k=3,
                    collection_name=col_name,
                )
                reference_texts = [
                    text.strip()
                    for c in chunks
                    if (text := c.get("content", c.get("text", ""))) and text.strip()
                ]
                if not reference_texts:
                    logger.warning(
                        "testset_gen_empty_references_skipped",
                        doc_id=doc["id"],
                        source=doc.get("source"),
                    )
                    continue
            except Exception as e:
                logger.warning(
                    "testset_gen_chunk_fetch_failed",
                    doc_id=doc["id"],
                    error=str(e),
                )
                reference_texts = []

            # 3. 构建 LLM prompt 生成 question + ground_truth
            doc_title = doc.get("title", "Untitled")
            doc_type_val = doc.get("doc_type", "txt")

            context_snippet = ""
            if reference_texts:
                context_snippet = "\n\n".join(t[:500] for t in reference_texts[:2])

            prompt = f"""你是一个 RAG 系统测试集生成器。根据以下文档信息生成一个高质量的问答对。

文档标题: {doc_title}
文档类型: {doc_type_val}

文档内容片段:
{context_snippet if context_snippet else "(无内容片段)"}

要求：
1. 问题应当多样化，从以下类型中选择一种（优先选择能体现文档特点的类型）：
   - 事实查询：询问文档中的具体数据、日期、名称、定义
   - 比较问题：询问两个或多个概念/选项的区别
   - 总结问题：需要综合多个段落的信息才能完整回答
2. ground_truth 应当全面准确，包含文档中的所有相关要点
3. ground_truth 应当包含具体的数字、名称等细节，而非笼统概括
4. 问题和答案都使用中文

高质量示例：
文档内容：公司年假政策——工龄1-5年10天，5-10年15天，10年以上20天。年假可累计至次年3月31日。
问题："工龄8年的员工每年有多少天年假？年假累计到什么时候？"
ground_truth："工龄5-10年的员工每年享有15天年假。年假可累计至次年3月31日，逾期未休则自动清零。"

输出格式（严格JSON）：
{{"question": "...", "ground_truth": "..."}}

只输出 JSON，不要其他内容。"""

            try:
                # LLMClient 包装了 AsyncOpenAI client
                if isinstance(llm_client, LLMClient):
                    response = await llm_client.client.chat.completions.create(
                        model=llm_client.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=500,
                    )
                    result_text = response.choices[0].message.content.strip()
                elif isinstance(llm_client, AsyncOpenAI):
                    # 兼容直接传入 AsyncOpenAI 实例
                    response = await llm_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=500,
                    )
                    result_text = response.choices[0].message.content.strip()
                else:
                    # 兼容 LangGraph / LangChain LLM 接口
                    response = await llm_client.ainvoke(prompt)
                    result_text = response.content.strip() if hasattr(response, "content") else str(response).strip()
            except Exception as e:
                logger.warning(
                    "testset_gen_llm_failed",
                    doc_id=doc["id"],
                    error=str(e),
                )
                continue

            # 解析 LLM 输出
            try:
                # 清理可能的前后缀
                if result_text.startswith("```"):
                    result_text = result_text.split("\n", 1)[1]
                    if result_text.endswith("```"):
                        result_text = result_text[:-3]
                obj = json.loads(result_text)
                question = obj.get("question", "")
                ground_truth = obj.get("ground_truth", "")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "testset_gen_parse_failed",
                    doc_id=doc["id"],
                    raw_output=result_text[:200],
                    error=str(e),
                )
                continue

            if _is_valid_generated_case(question, ground_truth, reference_texts):
                testset.append(TestsetItem(
                    question=question.strip(),
                    ground_truth=ground_truth.strip(),
                    reference_contexts=reference_texts,
                ))
            else:
                logger.warning(
                    "testset_gen_invalid_case_skipped",
                    doc_id=doc["id"],
                    question=question[:120],
                    has_ground_truth=bool(ground_truth),
                    reference_count=len(reference_texts),
                )

        logger.info("testset_generated", count=len(testset), requested=size)
        return testset

    except Exception as e:
        logger.error("testset_generate_failed", error=str(e))
        raise RuntimeError(f"测试集生成失败: {e}") from e
