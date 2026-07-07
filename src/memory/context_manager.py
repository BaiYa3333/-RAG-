"""Context window management — token estimation, history truncation, summarization."""

from __future__ import annotations

from src.config import settings
from src.memory.service import MemoryService
from src.utils.logger import logger


def estimate_tokens(text: str) -> int:
    """Chinese-aware token count estimator.

    CJK unified ideographs (U+4E00–U+9FFF) count as 1 token each.
    All other characters count at a 4:1 ratio (4 chars ≈ 1 token).
    Empty input returns 0.
    """
    if not text:
        return 0

    cjk_count = 0
    other_count = 0
    for ch in text:
        if '一' <= ch <= '鿿':
            cjk_count += 1
        else:
            other_count += 1

    # CJK: 1 char = 1 token; Other: 4 chars ≈ 1 token
    estimated = cjk_count + max(1, other_count // 4) if other_count > 0 else cjk_count
    return estimated


def estimate_history_tokens(history: list[dict]) -> int:
    """Estimate total tokens across conversation history."""
    total = 0
    for turn in history:
        total += estimate_tokens(turn.get("content", ""))
    return total


async def maybe_truncate_history(
    memory: MemoryService,
    session_id: str,
    llm_client=None,
) -> list[dict]:
    """Load history and truncate if it exceeds max_context_tokens.

    When truncation happens, old messages are summarized via LLM and
    the summary is prepended as a system message.
    """
    history = await memory.load_history(session_id)
    total_tokens = estimate_history_tokens(history)
    max_tokens = settings.max_context_tokens

    if total_tokens <= max_tokens:
        return history

    # Truncation needed: keep most recent messages under limit
    logger.info(
        "context_truncation_needed",
        session_id=session_id,
        total_tokens=total_tokens,
        max_tokens=max_tokens,
    )

    # Split into old (to summarize) and recent (to keep)
    kept: list[dict] = []
    kept_tokens = 0
    old: list[dict] = []

    for turn in reversed(history):
        turn_tokens = estimate_tokens(turn.get("content", ""))
        if kept_tokens + turn_tokens < max_tokens // 2:
            kept.insert(0, turn)
            kept_tokens += turn_tokens
        else:
            old.insert(0, turn)

    # Summarize old messages if there are any
    if old and llm_client:
        try:
            summary = await _summarize_messages(llm_client, old)
            if summary:
                await memory.update_summary(session_id, summary)
                kept.insert(0, {"role": "system", "content": f"[Previous conversation summary]: {summary}"})
        except Exception as e:
            logger.warning("history_summarization_failed", error=str(e))

    return kept


async def check_and_summarize(
    memory: MemoryService,
    session_id: str,
    llm_client=None,
) -> None:
    """Check if session exceeds summarize_after_turns threshold and summarize if needed."""
    turn_count = await memory.get_turn_count(session_id)
    threshold = settings.summarize_after_turns

    if turn_count >= threshold:
        logger.info(
            "auto_summarize_triggered",
            session_id=session_id,
            turn_count=turn_count,
            threshold=threshold,
        )
        await maybe_truncate_history(memory, session_id, llm_client)


async def _summarize_messages(llm_client, messages: list[dict]) -> str | None:
    """Use LLM to generate a concise summary of messages."""
    if not messages:
        return None

    conversation_text = "\n".join(
        f"{m['role']}: {m['content'][:500]}" for m in messages[-20:]
    )

    prompt = (
        "Summarize the following conversation in 2-3 sentences, preserving key facts and decisions:\n\n"
        f"{conversation_text}\n\nSummary:"
    )

    try:
        response = await llm_client.client.chat.completions.create(
            model=llm_client.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("llm_summarize_failed", error=str(e))
        return None
