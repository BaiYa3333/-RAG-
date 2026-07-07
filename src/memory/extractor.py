"""长期记忆提取器 — 从对话中自动提取值得记忆的用户信息."""

from __future__ import annotations

import json
import logging

from src.llm.factory import create_llm

logger = logging.getLogger(__name__)

MEMORY_EXTRACT_PROMPT = """分析以下对话，提取值得长期记忆的用户信息。
只提取明确陈述的事实，不要推测。

对话内容：
{conversation}

如有值得记忆的信息，以 JSON 数组返回（每项包含 type 和 content）：
[
  {{"type": "fact", "content": "用户名叫张三"}},
  {{"type": "preference", "content": "用户偏好简洁的回答风格"}},
  {{"type": "context", "content": "用户正在准备2026年度汇报"}}
]

type 说明：
- fact: 用户个人信息（姓名、职位、部门、技能等）
- preference: 用户偏好（风格、格式、语言等）
- context: 当前工作上下文（项目、任务、目标等）

如果没有值得记忆的信息，返回空数组：[]
只返回 JSON，不要其他内容。"""


async def extract_memories(
    conversation: list[dict],
    model_name: str | None = None,
) -> list[dict]:
    """从对话中提取值得记忆的信息片段。

    Args:
        conversation: 对话历史 [{role, content}, ...]
        model_name: LLM 模型名

    Returns:
        [{"type": "fact", "content": "用户名叫张三"}, ...]
    """
    if not conversation:
        return []

    # 只取最近的对话轮次（最多10轮），避免 token 超限
    recent = conversation[-20:] if len(conversation) > 20 else conversation
    conv_text = "\n".join(
        f"{m.get('role', 'unknown')}: {m.get('content', '')}"
        for m in recent
    )

    prompt = MEMORY_EXTRACT_PROMPT.format(conversation=conv_text)

    try:
        llm = create_llm(model_name=model_name)
        resp = await llm.client.chat.completions.create(
            model=llm.model_id,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
        # 移除可能的 markdown 代码块标记
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        return []
    except Exception as exc:
        logger.warning("memory_extract_failed: %s", exc)
        return []


async def extract_and_save_memories(
    conversation: list[dict],
    user_id: str,
    memory_service,
    model_name: str | None = None,
    session_id: str | None = None,
) -> int:
    """从对话提取记忆并保存到数据库。

    Args:
        conversation: 对话历史 [{role, content}, ...]
        user_id: 用户 ID
        memory_service: MemoryService 实例
        model_name: LLM 模型名
        session_id: 当前会话 ID（用于记忆隔离）

    Returns:
        保存的记忆条数
    """
    if not user_id or user_id == "anonymous":
        return 0

    memories = await extract_memories(conversation, model_name=model_name)
    if not memories:
        return 0

    saved_count = 0
    for mem in memories:
        try:
            await memory_service.save_user_memory(
                user_id=user_id,
                memory_type=mem.get("type", "fact"),
                content=mem.get("content", ""),
                session_id=session_id,
            )
            saved_count += 1
        except Exception as exc:
            logger.warning("memory_save_failed: %s", exc)

    if saved_count > 0:
        logger.info("memories_extracted_and_saved", user_id=user_id, count=saved_count, session_id=session_id)

    return saved_count
