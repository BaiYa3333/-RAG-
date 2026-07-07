from src.config import settings
from src.llm.providers.deepseek import deepseek_spec
from src.llm.providers.qwen import qwen_spec

MODEL_REGISTRY: dict = {}

_m = deepseek_spec()
MODEL_REGISTRY[_m.name] = _m

_m = qwen_spec()
MODEL_REGISTRY[_m.name] = _m

DEFAULT_MODEL: str = settings.default_llm_model
