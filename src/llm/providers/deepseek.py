from src.config import settings
from src.llm.providers.base import ModelSpec


def deepseek_spec() -> ModelSpec:
    return ModelSpec(
        name="deepseek-chat",
        provider="deepseek",
        model_id="deepseek-chat",
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
        default_params={"temperature": 0.7, "max_tokens": 4096, "top_p": 0.9},
        label="DeepSeek Chat",
    )
