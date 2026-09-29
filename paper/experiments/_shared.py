"""Shared code for the frozen paper experiments."""

from __future__ import annotations

import csv
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlm_core.io import read_jsonl, write_json
from vlm_core.schema import DiagnosticSample


def load_samples(data_dir: Path, split: str | None = None) -> list[DiagnosticSample]:
    split_by_id = None
    if split:
        split_by_id = {row["sample_id"]: row["split"] for row in read_jsonl(data_dir / "splits.jsonl")}

    samples = []
    for row in read_jsonl(data_dir / "samples.jsonl"):
        if split_by_id and split_by_id[row["sample_id"]] != split:
            continue
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            root_path = ROOT / image_path
            image_path = root_path if root_path.is_file() else data_dir / image_path
        samples.append(
            DiagnosticSample(
                sample_id=row["sample_id"],
                primitive=row["primitive"],
                image_path=str(image_path),
                question=row["question"],
                answer=row["answer"],
                subtask=row["subtask"],
                difficulty=row["difficulty"],
                choices=row["choices"],
                metadata=row["metadata"],
                counterfactual_id=row.get("counterfactual_id") or row.get("pair_id"),
                source=row["source"],
            )
        )
    return samples


def split_map(data_dir: Path) -> dict[str, str]:
    return {row["sample_id"]: row["split"] for row in read_jsonl(data_dir / "splits.jsonl")}


def prompt(question: str) -> str:
    return f'/no_think\n{question}\nRespond only as compact JSON: {{"answer": "<final answer>"}}'


def patch_prompt(question: str) -> str:
    return f"/no_think\n{question}\nAnswer:"


def token_masks(inputs, adapter, samples: list[DiagnosticSample], text_only: bool):
    return adapter.token_masks(inputs, [sample.question for sample in samples], text_only=text_only)


def extract_features(
    adapter,
    samples: list[DiagnosticSample],
    batch_size: int,
    text_only: bool,
    cache_dir: Path,
    cache_key: dict,
):
    import torch

    collected = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            shard_path = cache_dir / f"batch_{start:06d}.pt"
            expected_ids = [sample.sample_id for sample in batch]
            if shard_path.exists():
                shard = torch.load(shard_path, map_location="cpu", weights_only=False)
                if shard.get("cache_key") != cache_key or shard.get("sample_ids") != expected_ids:
                    raise SystemExit(f"Incompatible representation readout feature shard: {shard_path}")
                features = shard["features"]
            else:
                inputs = adapter.prepare_inputs(
                    [prompt(sample.question) for sample in batch],
                    [None if text_only else sample.image_path for sample in batch],
                    enable_thinking=False,
                )
                outputs = adapter.forward(inputs, output_hidden_states=True)
                features = {}
                for group, mask in token_masks(inputs, adapter, batch, text_only).items():
                    weights = mask.unsqueeze(-1)
                    features[group] = [
                        ((hidden * weights).sum(1) / weights.sum(1).clamp_min(1)).float().cpu()
                        for hidden in outputs.hidden_states
                    ]
                temporary = shard_path.with_suffix(".pt.tmp")
                torch.save({"cache_key": cache_key, "sample_ids": expected_ids, "features": features}, temporary)
                temporary.replace(shard_path)
                del outputs, inputs
                torch.cuda.empty_cache()
                print(f"[representation readout/{adapter.spec.key}] {cache_dir.name} {min(start + len(batch), len(samples))}/{len(samples)}", flush=True)
            for group, layers in features.items():
                collected.setdefault(group, [[] for _ in layers])
                for layer, value in enumerate(layers):
                    collected[group][layer].append(value)
    return {group: [torch.cat(parts) for parts in layers] for group, layers in collected.items()}


def probe_rows(
    features,
    samples: list[DiagnosticSample],
    labels: list[str],
    splits: dict[str, str],
    task: str,
    subset: str,
    ridge: float,
    seed: int,
    controls=("main", "shuffled_label", "random_label"),
):
    rows = []
    train = [i for i, sample in enumerate(samples) if splits[sample.sample_id] == "train"]
    test = [i for i, sample in enumerate(samples) if splits[sample.sample_id] == "test"]
    for control in controls:
        target = labels[:]
        rng = random.Random(seed + {"shuffled_label": 101, "random_label": 202}.get(control, 0))
        if control == "shuffled_label":
            rng.shuffle(target)
        elif control == "random_label":
            classes = sorted(set(target))
            target = [rng.choice(classes) for _ in target]
        majority = Counter(target[i] for i in train).most_common(1)[0][0]
        baseline = sum(target[i] == majority for i in test) / len(test)
        for group, layers in features.items():
            for layer, x in enumerate(layers):
                rows.append(
                    {
                        "task": task,
                        "subset": subset,
                        "control": control,
                        "token_group": group,
                        "layer": layer,
                        "train_n": len(train),
                        "test_n": len(test),
                        "accuracy": fit_ridge(x[train], x[test], [target[i] for i in train], [target[i] for i in test], ridge),
                        "majority_baseline": baseline,
                    }
                )
    return rows


def fit_ridge(x_train, x_test, train_labels, test_labels, ridge: float) -> float:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = sorted(set(train_labels))
    label_id = {label: i for i, label in enumerate(classes)}
    x_train, x_test = x_train.to(device), x_test.to(device)
    mean, std = x_train.mean(0, keepdim=True), x_train.std(0, keepdim=True).clamp_min(1e-6)
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    y_train = torch.tensor([label_id[label] for label in train_labels], device=device)
    y_test = torch.tensor([label_id[label] for label in test_labels], device=device)
    onehot = torch.nn.functional.one_hot(y_train, len(classes)).to(x_train.dtype)
    kernel = x_train @ x_train.T + ridge * torch.eye(len(x_train), device=device)
    scores = x_test @ x_train.T @ torch.linalg.solve(kernel, onehot)
    return round((scores.argmax(1) == y_test).float().mean().item(), 6)


def write_rows(result_dir: Path, name: str, rows: list[dict]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_dir / f"{name}.json", rows)
    with (result_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
