"""Shared state-prediction and evaluation helpers for the main training chain."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .io import read_jsonl
from .models import build_vision_message


FROZEN_LAYER = 20
FROZEN_TARGET_SITE = "answer_suffix"
TARGET_SUFFIXES = {
    "last_prompt": "",
    "answer_suffix": " Therefore, the answer is",
    "dummy_thinking": " We compare the two sides briefly. Therefore, the answer is",
}


class StatePredictor(nn.Module):
    def __init__(self, d_model: int, kind: str, architecture: str, bottleneck: int | None = None):
        super().__init__()
        self.kind = kind
        self.architecture = architecture
        if architecture == "linear":
            self.net = nn.Linear(d_model, d_model)
            self.gate = None
        else:
            if bottleneck is None:
                raise ValueError("MLP requires a bottleneck.")
            self.net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, d_model),
            )
            self.gate = nn.Parameter(torch.tensor(0.01)) if kind == "residual" else None

    def forward(self, h_no: torch.Tensor) -> torch.Tensor:
        output = self.net(h_no)
        return output * self.gate if self.gate is not None else output


class TargetTransform:
    def __init__(self, mean_value: torch.Tensor, scale: torch.Tensor, enabled: bool):
        self.mean = mean_value
        self.scale = scale
        self.enabled = enabled

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.scale if self.enabled else value

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale + self.mean if self.enabled else value

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.cpu(), "scale": self.scale.cpu(), "enabled": self.enabled}

    @classmethod
    def from_state(cls, state: dict[str, Any], device: torch.device) -> "TargetTransform":
        return cls(state["mean"].to(device), state["scale"].to(device), bool(state["enabled"]))


def decoder_layers(model):
    paths = (
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "model", "layers"),
        ("model", "layers"),
        ("language_model", "layers"),
    )
    for path in paths:
        value = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if isinstance(value, nn.ModuleList) and len(value) > 0:
            return value
    raise ValueError("Could not find decoder layers.")


def find_final_norm(model) -> nn.Module:
    paths = (
        ("model", "language_model", "norm"),
        ("language_model", "model", "norm"),
        ("model", "model", "norm"),
        ("model", "norm"),
        ("language_model", "norm"),
    )
    for path in paths:
        value = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if isinstance(value, nn.Module):
            return value
    raise ValueError("Could not find final decoder norm.")


def tokenize_generation(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(encoded["input_ids"]), [tuple(offset) for offset in encoded["offset_mapping"]]


def indices_overlapping(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    if end <= start:
        return []
    return [index for index, (left, right) in enumerate(offsets) if right > start and left < end]


def padded_generation(prompt, tokenized, tokenizer):
    max_len = max(len(ids) for ids, _offsets in tokenized)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_ids = torch.full((len(tokenized), max_len), pad_id, dtype=prompt["input_ids"].dtype)
    gen_mask = torch.zeros((len(tokenized), max_len), dtype=prompt["attention_mask"].dtype)
    for index, (ids, _offsets) in enumerate(tokenized):
        gen_ids[index, : len(ids)] = torch.tensor(ids, dtype=gen_ids.dtype)
        gen_mask[index, : len(ids)] = 1
    return gen_ids, gen_mask


def extend_inputs(prompt, gen_ids, gen_mask, device):
    inputs = {key: value.to(device) for key, value in prompt.items()}
    prompt_len = inputs["input_ids"].shape[1]
    inputs["input_ids"] = torch.cat([inputs["input_ids"], gen_ids.to(device)], dim=1)
    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], gen_mask.to(device)], dim=1)
    for key, value in list(inputs.items()):
        if key in {"input_ids", "attention_mask", "pixel_values"} or value.ndim != 2 or value.shape[1] != prompt_len:
            continue
        extension = torch.zeros((value.shape[0], gen_ids.shape[1]), dtype=value.dtype, device=device)
        inputs[key] = torch.cat([value, extension], dim=1)
    return inputs


def append_suffix(inputs, suffix_ids: list[int], tokenizer):
    if not suffix_ids:
        return inputs
    batch = inputs["input_ids"].shape[0]
    ids = torch.tensor(suffix_ids, dtype=inputs["input_ids"].dtype).repeat(batch, 1)
    mask = torch.ones((batch, len(suffix_ids)), dtype=inputs["attention_mask"].dtype)
    output = dict(inputs)
    prompt_len = output["input_ids"].shape[1]
    output["input_ids"] = torch.cat([output["input_ids"], ids], dim=1)
    output["attention_mask"] = torch.cat([output["attention_mask"], mask], dim=1)
    for key, value in list(output.items()):
        if key in {"input_ids", "attention_mask"} or value.ndim != 2 or value.shape[1] != prompt_len:
            continue
        extension = torch.zeros((batch, len(suffix_ids)), dtype=value.dtype)
        output[key] = torch.cat([value, extension], dim=1)
    return output


def compositional_no_thinking_prompt(row: dict[str, Any]) -> str:
    choices = ", ".join(row["answer_candidates"])
    return (
        "Look at the image and consider the red and blue target shapes. "
        "The gray shapes are occluders, not target objects, so do not count them.\n"
        f"{row['question']}\n"
        "/no_think\n"
        f"Answer only one of: {choices}."
    )


def no_think_inputs(bundle, rows: list[dict[str, Any]], target_site: str):
    prompts = bundle.processor.apply_chat_template(
        [build_vision_message(compositional_no_thinking_prompt(row), row["image_path"]) for row in rows],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    suffix = TARGET_SUFFIXES[target_site]
    if suffix:
        prompts = append_suffix(prompts, bundle.tokenizer(suffix, add_special_tokens=False).input_ids, bundle.tokenizer)
    inputs = {key: value.to(bundle.model.device) for key, value in prompts.items()}
    positions = inputs["attention_mask"].shape[1] - 1 - torch.flip(inputs["attention_mask"], dims=[1]).argmax(dim=1)
    if not bool(torch.all(positions == inputs["input_ids"].shape[1] - 1)):
        raise ValueError("No-thinking answer-site batches must have aligned final live positions.")
    return inputs, positions.to(bundle.model.device)


def capture_resid_pre(bundle, inputs, positions, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    captured = {}

    def hook(_module, layer_inputs):
        hidden = layer_inputs[0]
        row_ids = torch.arange(positions.shape[0], device=hidden.device)
        captured["hidden"] = hidden[row_ids, positions.to(hidden.device)].detach().float().cpu()

    handle = decoder_layers(bundle.model)[layer_index].register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            outputs = bundle.model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    return captured["hidden"], outputs.logits[:, -1]


def patched_logits(bundle, inputs, positions, layer_index: int, replacement: torch.Tensor) -> torch.Tensor:
    layer = decoder_layers(bundle.model)[layer_index]

    def hook(_module, layer_inputs):
        hidden = layer_inputs[0].clone()
        row_ids = torch.arange(hidden.shape[0], device=hidden.device)
        hidden[row_ids, positions.to(hidden.device)] = replacement.to(hidden.device, dtype=hidden.dtype)
        return (hidden, *layer_inputs[1:])

    handle = layer.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            outputs = bundle.model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    return outputs.logits[:, -1]


def differentiable_patched_logits(bundle, inputs, positions, layer_index: int, replacement: torch.Tensor) -> torch.Tensor:
    layer = decoder_layers(bundle.model)[layer_index]

    def hook(_module, layer_inputs):
        hidden = layer_inputs[0].clone()
        row_ids = torch.arange(hidden.shape[0], device=hidden.device)
        hidden[row_ids, positions.to(hidden.device)] = replacement.to(hidden.device, dtype=hidden.dtype)
        return (hidden, *layer_inputs[1:])

    handle = layer.register_forward_pre_hook(hook)
    try:
        outputs = bundle.model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    return outputs.logits[:, -1]


def make_transform(target: torch.Tensor, threshold: float) -> tuple[TargetTransform, dict[str, float]]:
    mean_value = target.mean(dim=0, keepdim=True)
    scale = target.std(dim=0, keepdim=True).clamp_min(1e-6)
    positive = scale.flatten()[scale.flatten() > 1e-6]
    p05 = torch.quantile(positive, 0.05).item() if positive.numel() else 1.0
    p95 = torch.quantile(positive, 0.95).item() if positive.numel() else 1.0
    dispersion = p95 / max(p05, 1e-12)
    enabled = dispersion > threshold
    if not enabled:
        mean_value = torch.zeros_like(mean_value)
        scale = torch.ones_like(scale)
    return TargetTransform(mean_value, scale, enabled), {
        "p05_std": p05,
        "p95_std": p95,
        "dispersion": dispersion,
        "enabled": enabled,
    }


def predicted_state(model: StatePredictor, transform: TargetTransform, h_no: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    decoded = transform.decode(model(h_no))
    return (h_no + decoded, decoded) if model.kind == "residual" else (decoded, decoded - h_no)


def state_loss(
    model: StatePredictor,
    transform: TargetTransform,
    h_no: torch.Tensor,
    target: torch.Tensor,
    lambda_cos: float,
) -> torch.Tensor:
    prediction_encoded = model(h_no)
    target_encoded = transform.encode(target)
    regression = F.smooth_l1_loss(prediction_encoded, target_encoded)
    prediction = transform.decode(prediction_encoded)
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1, eps=1e-8).mean()
    return regression + lambda_cos * cosine


def save_predictor(path: Path, model: StatePredictor, transform: TargetTransform, extra: dict[str, Any]) -> None:
    bottleneck = model.net[1].out_features if model.architecture == "mlp" else None
    torch.save(
        {
            "kind": model.kind,
            "architecture": model.architecture,
            "d_model": model.net.in_features if model.architecture == "linear" else model.net[1].in_features,
            "bottleneck": bottleneck,
            "model_state": model.state_dict(),
            "transform": transform.state_dict(),
            **extra,
        },
        path,
    )


def load_predictor(path: Path, device: torch.device) -> tuple[StatePredictor, TargetTransform, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = StatePredictor(
        checkpoint["d_model"],
        checkpoint["kind"],
        checkpoint["architecture"],
        checkpoint.get("bottleneck"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    transform = TargetTransform.from_state(checkpoint["transform"], device)
    return model, transform, checkpoint


def freeze_vlm(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def load_split(data_dir: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = torch.load(data_dir / f"{split}.pt", map_location="cpu", weights_only=True)
    metadata = read_jsonl(data_dir / f"{split}_metadata.jsonl")
    if payload["h_no"].shape[0] != len(metadata):
        raise SystemExit(f"Tensor/metadata mismatch for {split}.")
    return payload, metadata


def program_balanced_indices(metadata: list[dict[str, Any]], seed: int) -> list[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        grouped[row["program_id"]].append(index)
    if not grouped:
        return []
    rng = random.Random(seed)
    programs = sorted(grouped)
    target = max(len(indices) for indices in grouped.values())
    expanded = {}
    for program in programs:
        draws = []
        while len(draws) < target:
            cycle = grouped[program].copy()
            rng.shuffle(cycle)
            draws.extend(cycle)
        expanded[program] = draws[:target]
    order = []
    for offset in range(target):
        program_order = programs.copy()
        rng.shuffle(program_order)
        order.extend(expanded[program][offset] for program in program_order)
    return order


def candidate_score_matrix(tokenizer, logits: torch.Tensor, rows: list[dict[str, Any]]) -> torch.Tensor:
    width = max(len(row["answer_candidates"]) for row in rows)
    matrix = logits.new_full((len(rows), width), float("-inf"))
    for row_index, row in enumerate(rows):
        for candidate_index, candidate in enumerate(row["answer_candidates"]):
            token_ids = set()
            for text in (str(candidate), f" {candidate}", str(candidate).lower(), f" {str(candidate).lower()}"):
                encoded = tokenizer(text, add_special_tokens=False).input_ids
                if encoded:
                    token_ids.add(int(encoded[-1]))
            if not token_ids:
                raise ValueError(f"No token IDs for candidate {candidate!r}.")
            matrix[row_index, candidate_index] = logits[row_index, sorted(token_ids)].max()
    return matrix


def correct_margins(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    row_ids = torch.arange(scores.shape[0], device=scores.device)
    correct = scores[row_ids, labels]
    masked = scores.clone()
    masked[row_ids, labels] = float("-inf")
    return correct - masked.max(dim=-1).values


def mmstar_inputs(bundle, rows: list[dict[str, Any]], target_site: str):
    messages = []
    for row in rows:
        prompt = f"{row['question']}\n/no_think\nAnswer only A, B, C, or D."
        image = row["image"].convert("RGB")
        messages.append(
            [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        )
    prompts = bundle.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    suffix = TARGET_SUFFIXES[target_site]
    if suffix:
        prompts = append_suffix(prompts, bundle.tokenizer(suffix, add_special_tokens=False).input_ids, bundle.tokenizer)
    inputs = {key: value.to(bundle.model.device) for key, value in prompts.items()}
    positions = inputs["attention_mask"].shape[1] - 1 - torch.flip(inputs["attention_mask"], dims=[1]).argmax(dim=1)
    return inputs, positions.to(bundle.model.device)


def summarize_mmstar(rows: list[dict[str, Any]], expected: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], "overall")].append(row)
        groups[(row["condition"], row["category"])].append(row)
    output = []
    for (condition, category), values in sorted(groups.items()):
        if category == "overall" and len(values) != expected:
            raise SystemExit(f"Incomplete MMStar {condition}: {len(values)}/{expected}.")
        output.append(
            {
                "condition": condition,
                "category": category,
                "n": len(values),
                "accuracy": round(sum(row["is_correct"] for row in values) / len(values), 6),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def training_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def batches(values: Iterable[int], batch_size: int) -> Iterable[list[int]]:
    values = list(values)
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
