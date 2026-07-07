"""Chunk Refinement — LLM 驱动的文本去噪精炼.

对每个 chunk 调用 LLM 去除 PDF/DOCX 解析噪音（页眉页脚、页码、水印、格式残留），
同时保留所有事实信息、Markdown 表格结构和 [IMAGE:xxx] 占位符。

使用 asyncio.Semaphore 控制并发，LLM 失败时降级为保留原始文本。
"""

from __future__ import annotations

import asyncio
import time

from src.config import settings
from src.llm.factory import create_llm
from src.observability.langfuse_context import langfuse_context
from src.utils.logger import logger

# ── Chunk Refinement System Prompt (中文) ──────────────────────
REFINEMENT_SYSTEM_PROMPT = """你是一个专业的文档清洗助手。你的任务是对给定的文本片段进行去噪精炼，去除文档解析过程中引入的噪音，但严格保留所有事实信息。

## 需要去除的噪音类型
- 页眉、页脚、页码（如 "第 1 页"、"Page 1"、"1/10"）
- 水印文字（如 "机密"、"Draft"、"CONFIDENTIAL"）
- 格式残留（如多余的空格、换行符、制表符、HTML 标签残留）
- PDF 转换产生的乱码字符或重复标点
- 目录条目和交叉引用（如 "详见第 X 页"）
- 扫描/OCR 产生的边缘噪点文字

## 必须保留的内容
- 所有事实信息、数据、日期、数字
- Markdown 表格结构（|...| 格式）— 完全保留
- [IMAGE:xxx] 占位符 — 完全保留，不修改
- 标题层级（#、##、###）
- 列表结构（-、*、1.）
- 代码块（```...```）
- 所有专有名词、技术术语

## 重要约束
- 不要添加任何新信息
- 不要总结或改写内容
- 不要修改 Markdown 表格的任何一个字符
- 不要删除任何可能是正文内容的句子
- 如果文本已经很干净，直接原样返回

请直接返回清洗后的文本，不要添加任何解释或前缀。"""


class ChunkRefiner:
    """LLM 驱动的 Chunk 文本精炼器.

    对每个 chunk 调用 LLM 去除噪音，使用 asyncio.Semaphore 控制并发。
    LLM 调用失败时降级为保留原始文本，不阻塞管道。
    """

    def __init__(self, concurrency: int | None = None, llm_model: str | None = None):
        self.concurrency = concurrency or settings.rag_ingestion_enrichment_concurrency or 5
        self.llm_model = llm_model or settings.rag_ingestion_llm_model or "deepseek-chat"

    async def refine_chunks(
        self, chunks: list[dict], llm_model: str | None = None,
    ) -> list[dict]:
        """对 chunks 进行 LLM 去噪精炼.

        Args:
            chunks: chunk 列表 [{"content": ..., "metadata": {...}}, ...]
            llm_model: 可选覆盖默认 LLM 模型

        Returns:
            精炼后的 chunk 列表。失败 chunk 保留原始内容。
        """
        if not chunks:
            return chunks

        model = llm_model or self.llm_model
        t0 = time.monotonic()
        semaphore = asyncio.Semaphore(self.concurrency)

        async def refine_with_semaphore(chunk: dict) -> dict:
            async with semaphore:
                refined_text = await self._refine_one(chunk, model)
                if refined_text is not None and refined_text.strip():
                    return {**chunk, "content": refined_text}
                return chunk

        tasks = [refine_with_semaphore(c) for c in chunks]
        refined = await asyncio.gather(*tasks)

        success_count = sum(
            1 for orig, ref in zip(chunks, refined)
            if ref["content"] != orig["content"]
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "chunk_refinement_done",
            total=len(chunks),
            refined=success_count,
            failed=len(chunks) - success_count,
            elapsed_ms=round(elapsed_ms, 2),
        )

        # Langfuse tracing
        try:
            langfuse_context.update_current_observation(
                refinement_total=len(chunks),
                refinement_success=success_count,
                refinement_failed=len(chunks) - success_count,
                refinement_model=model,
                refinement_elapsed_ms=round(elapsed_ms, 2),
            )
        except Exception:
            pass

        return refined

    async def _refine_one(self, chunk: dict, model: str) -> str | None:
        """异步精炼单个 chunk."""
        content = chunk.get("content", "")
        if not content or not content.strip():
            return content

        try:
            llm = create_llm(model)
            response = await llm.client.chat.completions.create(
                model=llm.model_id,
                messages=[
                    {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.1,
                max_tokens=min(len(content) * 2, 4096),
                **{k: v for k, v in llm.default_params.items()
                   if k not in ("temperature", "max_tokens")},
            )
            refined = response.choices[0].message.content
            if refined and refined.strip():
                return refined.strip()
            return content  # LLM 返回空，降级为原文
        except Exception:
            chunk_id = chunk.get("chunk_id", chunk.get("metadata", {}).get("chunk_id", "?"))
            logger.warning("chunk_refinement_failed", chunk_id=chunk_id)
            return None  # 返回 None 表示降级，调用方保留原文
