from src.config import settings
from src.llm.providers.base import ModelSpec


def qwen_spec() -> ModelSpec:
    return ModelSpec(
        name="qwen3.6-plus",
        provider="qwen",
        model_id="qwen-plus",  # DashScope model ID — qwen-plus always maps to latest Qwen 3.6 Plus
        base_url=settings.qwen_base_url,
        api_key=settings.qwen_api_key,
        default_params={"temperature": 0.7, "max_tokens": 4096, "top_p": 0.9},
        label="Qwen 3.6 Plus",
    )
