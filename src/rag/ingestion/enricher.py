"""Metadata Enrichment — LLM 驱动的 Chunk 元数据自动生成.

对每个 chunk 调用 LLM 生成 Title (≤150 chars)、Summary (≤500 chars) 和 Tags (3-10 个)，
Title+Summary 拼接到 chunk content 前缀以参与 Embedding，Tags 存入 chunk.metadata。

使用 asyncio.Semaphore 控制并发，LLM 失败时降级为跳过。
"""

from __future__ import annotations

import asyncio
import json
import re
import time

from src.config import settings
from src.llm.factory import create_llm
from src.observability.langfuse_context import langfuse_context
from src.utils.logger import logger

# ── Metadata Enrichment System Prompt (中文) ───────────────────
ENRICHMENT_SYSTEM_PROMPT = """你是一个专业的文档分析与标注助手。你的任务是为给定的文本片段生成结构化的元数据，包括标题（Title）、摘要（Summary）和标签（Tags）。

## 输出格式
你必须严格输出 JSON 格式，不要包含任何其他文字：
```json
{
  "title": "文本片段的精炼标题（≤150字）",
  "summary": "文本片段的核心内容摘要（2-3句话，≤500字）",
  "tags": ["标签1", "标签2", "标签3"]
}
```

## 标题 (Title) 要求
- 精确概括该文本片段的主题
- 长度 ≤ 150 个字符
- 使用中文
- 如果是表格内容，标题应描述表格包含的数据类型

## 摘要 (Summary) 要求
- 2-3 句话提炼核心信息
- 长度 ≤ 500 个字符
- 保留关键数字、日期、专有名词
- 如果是表格数据，概括表格的主要维度和关键数据

## 标签 (Tags) 要求
- 3-10 个标签
- 优先提取文本中出现的专有名词、技术术语、领域概念
- 使用中文或英文（与原文保持一致）
- 从宽泛到具体排列

## 重要约束
- 不要编造原文中没有的信息
- 标签不要过于泛化（如 "文档"、"文本"）
- 如果文本已经是标题行或极其简短，title 可以直接使用原文首句"""


class MetadataEnricher:
    """LLM 驱动的 Chunk 元数据丰富器.

    为每个 chunk 生成 Title/Summary/Tags:
    - Title 和 Summary 拼接到 chunk content 前缀参与 Embedding
    - Tags 存入 chunk.metadata["tags"]

    LLM 调用失败时降级为跳过，保留原始 chunk 不变。
    """

    def __init__(self, concurrency: int | None = None, llm_model: str | None = None):
        self.concurrency = concurrency or settings.rag_ingestion_enrichment_concurrency or 5
        self.llm_model = llm_model or settings.rag_ingestion_llm_model or "deepseek-chat"

    async def enrich_chunks(
        self, chunks: list[dict], llm_model: str | None = None,
    ) -> list[dict]:
        """对 chunks 进行 LLM 元数据生成和内容增强.

        Args:
            chunks: chunk 列表 [{"content": ..., "metadata": {...}}, ...]
            llm_model: 可选覆盖默认 LLM 模型

        Returns:
            增强后的 chunk 列表。失败 chunk 保留原始内容不变。
        """
        if not chunks:
            return chunks

        model = llm_model or self.llm_model
        t0 = time.monotonic()
        semaphore = asyncio.Semaphore(self.concurrency)

        async def enrich_with_semaphore(chunk: dict) -> dict:
            async with semaphore:
                result = await self._enrich_one(chunk, model)
                return result if result is not None else chunk

        tasks = [enrich_with_semaphore(c) for c in chunks]
        enriched = await asyncio.gather(*tasks)

        success_count = sum(
            1 for orig, enr in zip(chunks, enriched)
            if enr.get("content") != orig.get("content")
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "metadata_enrichment_done",
            total=len(chunks),
            enriched=success_count,
            failed=len(chunks) - success_count,
            elapsed_ms=round(elapsed_ms, 2),
        )

        # Langfuse tracing
        try:
            langfuse_context.update_current_observation(
                enrichment_total=len(chunks),
                enrichment_success=success_count,
                enrichment_failed=len(chunks) - success_count,
                enrichment_model=model,
                enrichment_elapsed_ms=round(elapsed_ms, 2),
            )
        except Exception:
            pass

        return enriched

    async def _enrich_one(self, chunk: dict, model: str) -> dict | None:
        """异步增强单个 chunk."""
        content = chunk.get("content", "")
        if not content or not content.strip():
            return None

        try:
            llm = create_llm(model)
            response = await llm.client.chat.completions.create(
                model=llm.model_id,
                messages=[
                    {"role": "system", "content": ENRICHMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.3,
                max_tokens=800,
                **{k: v for k, v in llm.default_params.items()
                   if k not in ("temperature", "max_tokens")},
            )
            raw_output = response.choices[0].message.content
            if not raw_output or not raw_output.strip():
                return None

            metadata = self._parse_json_output(raw_output)
            if metadata is None:
                return None

            title = (metadata.get("title") or "").strip()
            summary = (metadata.get("summary") or "").strip()
            tags = metadata.get("tags") or []

            # Validate tags
            if not isinstance(tags, list):
                tags = []
            tags = [str(t).strip() for t in tags if str(t).strip()][:10]

            # Prepend Title + Summary to content for embedding
            original_content = content
            prefix_parts = []
            if title:
                prefix_parts.append(f"# {title}")
            if summary:
                prefix_parts.append(summary)
            enriched_content = original_content
            if prefix_parts:
                enriched_content = "\n\n".join(prefix_parts) + "\n\n" + original_content

            enriched_meta = {**chunk.get("metadata", {})}
            enriched_meta["tags"] = tags
            enriched_meta["enrichment_title"] = title
            enriched_meta["enrichment_summary"] = summary

            return {
                **chunk,
                "content": enriched_content,
                "metadata": enriched_meta,
            }
        except Exception:
            chunk_id = chunk.get("chunk_id", chunk.get("metadata", {}).get("chunk_id", "?"))
            logger.warning("metadata_enrichment_failed", chunk_id=chunk_id)
            return None

    @staticmethod
    def _parse_json_output(raw: str) -> dict | None:
        """从 LLM 原始输出中解析 JSON 元数据."""
        # Try direct JSON parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from code blocks
        if "```json" in raw:
            start = raw.find("```json") + 7
            end = raw.find("```", start)
            if end > start:
                try:
                    return json.loads(raw[start:end].strip())
                except json.JSONDecodeError:
                    pass
        elif "```" in raw:
            start = raw.find("```") + 3
            end = raw.find("```", start)
            if end > start:
                try:
                    return json.loads(raw[start:end].strip())
                except json.JSONDecodeError:
                    pass

        # Try finding JSON object with regex
        try:
            match = re.search(r'\{[^{}]*"title"[^{}]*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        logger.warning("enrichment_json_parse_failed", raw_preview=raw[:200])
        return None
