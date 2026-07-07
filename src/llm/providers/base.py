from dataclasses import dataclass, field


def _default_label(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").title()


@dataclass
class ModelSpec:
    name: str
    provider: str
    model_id: str
    base_url: str
    api_key: str
    default_params: dict = field(default_factory=lambda: {
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 0.9,
    })
    label: str = ""

    def display_label(self) -> str:
        return self.label or _default_label(self.name)
