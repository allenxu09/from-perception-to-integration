"""Small data containers for experiments."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DiagnosticSample:
    sample_id: str
    primitive: str
    image_path: str | None
    question: str
    answer: str
    subtask: str = "base"
    difficulty: str = "base"
    choices: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    counterfactual_id: str | None = None
    source: str = "controlled_core"


@dataclass(frozen=True)
class ModelBundle:
    name: str
    model: Any
    processor: Any | None = None
    tokenizer: Any | None = None
    device: str = "auto"


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    answer: str
    raw_output: str
    score: float | None = None


def image_path_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("type") == "image":
            return str(message["value"])
    raise ValueError("MMStar manifest row has no image message.")
