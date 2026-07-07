from dataclasses import dataclass, field

from openai import AsyncOpenAI

from src.llm.registry import DEFAULT_MODEL, MODEL_REGISTRY

# 模块级 AsyncOpenAI client 缓存 — 按 (base_url, api_key_prefix) 复用 httpx 连接池
_client_cache: dict[str, AsyncOpenAI] = {}


def _client_cache_key(api_key: str, base_url: str) -> str:
    """生成 client 缓存键（基于连接参数）。"""
    return f"{base_url}:{api_key[:8]}"


@dataclass
class LLMClient:
    client: AsyncOpenAI
    model_id: str
    default_params: dict = field(default_factory=dict)


def create_llm(model_name: str | None = None, **overrides) -> LLMClient:
    name = model_name or DEFAULT_MODEL

    spec = MODEL_REGISTRY.get(name)
    if spec is None:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    merged_params = {**spec.default_params, **overrides}

    cache_key = _client_cache_key(spec.api_key, spec.base_url)
    if cache_key not in _client_cache:
        _client_cache[cache_key] = AsyncOpenAI(
            api_key=spec.api_key,
            base_url=spec.base_url,
        )

    return LLMClient(
        client=_client_cache[cache_key],
        model_id=spec.model_id,
        default_params=merged_params,
    )
