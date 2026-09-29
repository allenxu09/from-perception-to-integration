"""Frozen answer-state dynamics: decodability versus usability across native reasoning."""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlm_core.io import append_jsonl, read_jsonl, write_json  # noqa: E402
from _model_adapter import ModelAdapter, model_spec, prompt_fingerprint  # noqa: E402


DATA_DIR = ROOT / "paper/data/occluded_target_reasoning_v2"
PROMPT = (
    "Look at the image. Think carefully about the occluded shapes, their colors, and which side they are on.\n"
    "Which side has more red circles after completing the occluded shapes?\n"
    "A. left\n"
    "B. right\n"
    "Think step by step."
)
PROMPT_PROTOCOL = "native_thinking_v2"
PROMPT_OFFICIAL_SWITCH = (
    "Look at the image. Think carefully about the occluded shapes, their colors, and which side they are on.\n"
    "Which side has more red circles after completing the occluded shapes?\n"
    "A. left\n"
    "B. right"
)
ANALYSIS_PROTOCOL = "fixed_validation_layer_pair_complete_v2"
STATES = ("pre_think", "early_think", "middle_think", "late_think", "pre_answer")
SUFFIX = " Therefore, the answer is"
SEED = 13
VALIDATION_PAIRS = 80
RIDGE = 100.0
BOOTSTRAP_ROUNDS = 1000
CHECKPOINT_EVERY = 50


def main(model_key: str = "qwen") -> None:
    spec = model_spec(model_key)
    experiment_prompt = PROMPT_OFFICIAL_SWITCH if spec.quantized else PROMPT
    result_dir = ROOT / "paper/results" / spec.result_slug / "decodability_usability_gap"
    trace_path = result_dir / "native_thinking_traces.jsonl"
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    samples = load_samples()
    traces = load_traces(samples, trace_path)
    splits = protocol_splits(samples)
    sample_ids = [row["sample_id"] for row in samples]
    cache_key = {
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["version"],
        "dataset_fingerprint": manifest["fingerprint"],
        "sample_ids": sample_ids,
        "model": spec.model_id,
        "model_revision": spec.revision,
        "prompt": experiment_prompt,
        "prompt_fingerprint": prompt_fingerprint(experiment_prompt),
        "prompt_protocol": "thinking_switch_v3" if spec.quantized else PROMPT_PROTOCOL,
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "split": "canonical test; 80 canonical-train pairs reserved for validation",
    }
    result_dir.mkdir(parents=True, exist_ok=True)

    adapter = ModelAdapter.load(model_key)
    cache_key["environment"] = adapter.provenance()
    check_or_write_json(result_dir / "provenance.json", cache_key)
    rows, excluded = pair_complete_rows(samples, traces)
    eligible_ids = {row["sample_id"] for row in rows}
    split_ids = {
        name: [row["sample_id"] for row in rows if splits[row["sample_id"]] == name]
        for name in ("train", "validation", "test")
    }
    split_pairs = {
        name: sorted({row["pair_id"] for row in rows if splits[row["sample_id"]] == name})
        for name in split_ids
    }
    assert all(len(split_ids[name]) == 2 * len(split_pairs[name]) for name in split_ids)

    prethink = collect_prethink_layers(adapter, rows, cache_key, result_dir)
    layer_rows, selected_layer = select_layer(prethink, rows, split_ids)
    fixed = collect_fixed_layer_states(adapter, rows, selected_layer, cache_key, result_dir)
    suffix = collect_suffix_margins(adapter, rows, cache_key, result_dir)
    metrics = evaluate(rows, split_ids, split_pairs["test"], fixed, suffix, selected_layer)

    write_json(result_dir / "layer_selection.json", layer_rows)
    write_json(result_dir / "metrics.json", metrics)
    write_csv(result_dir / "metrics.csv", metrics["states"])
    write_json(
        result_dir / "run_summary.json",
        {
            "selected_layer": selected_layer,
            "eligible_samples": len(eligible_ids),
            "eligible_pairs": len(eligible_ids) // 2,
            "excluded_pairs": excluded,
            "split_samples": {name: len(ids) for name, ids in split_ids.items()},
            "split_pairs": {name: len(ids) for name, ids in split_pairs.items()},
            "test_sample_ids": split_ids["test"],
            "model": spec.model_id,
            "model_revision": spec.revision,
            "precision": "fp8_dynamic_w8a8" if spec.fp8_dynamic else ("bitsandbytes_int8" if spec.load_in_8bit else "bfloat16"),
            "output_path": str(result_dir / "metrics.json"),
        },
    )
    print(f"[state dynamics] complete: layer={selected_layer}, test_pairs={len(split_pairs['test'])}", flush=True)


def load_samples() -> list[dict]:
    split_by_id = {row["sample_id"]: row["split"] for row in read_jsonl(DATA_DIR / "splits.jsonl")}
    samples = []
    for row in read_jsonl(DATA_DIR / "samples.jsonl"):
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = DATA_DIR / image_path
        samples.append(
            {
                "sample_id": str(row["sample_id"]),
                "pair_id": str(row["pair_id"]),
                "pair_side": str(row["metadata"]["pair_side"]),
                "canonical_split": split_by_id[row["sample_id"]],
                "image_path": str(image_path),
                "answer": str(row["answer"]),
                "larger_target_side": str(row["metadata"]["larger_target_side"]),
            }
        )
    return samples


def load_traces(samples: list[dict], trace_path: Path) -> dict[str, dict]:
    if not trace_path.exists():
        raise SystemExit("Run paper/experiments/native_thinking_generation.py first")
    expected = {row["sample_id"]: row for row in samples}
    traces = {}
    for row in read_jsonl(trace_path):
        sample_id = str(row["sample_id"])
        if sample_id not in expected:
            raise SystemExit(f"Unknown trace sample: {sample_id}")
        if sample_id in traces:
            raise SystemExit(f"Duplicate trace sample: {sample_id}")
        sample = expected[sample_id]
        if row["pair_id"] != sample["pair_id"] or row["answer"] != sample["answer"]:
            raise SystemExit(f"Trace/sample mismatch: {sample_id}")
        traces[sample_id] = row
    if len(traces) != len(samples):
        raise SystemExit(f"Incomplete native traces: {len(traces)}/{len(samples)}")
    return traces


def protocol_splits(samples: list[dict]) -> dict[str, str]:
    canonical_train = sorted({row["pair_id"] for row in samples if row["canonical_split"] == "train"})
    rng = random.Random(SEED)
    rng.shuffle(canonical_train)
    validation = set(canonical_train[:VALIDATION_PAIRS])
    return {
        row["sample_id"]: "test" if row["canonical_split"] == "test" else "validation" if row["pair_id"] in validation else "train"
        for row in samples
    }


def pair_complete_rows(samples: list[dict], traces: dict[str, dict]) -> tuple[list[dict], int]:
    by_pair = defaultdict(list)
    for sample in samples:
        trace = traces[sample["sample_id"]]
        row = {**sample, **trace}
        row["token_data"] = trace.get("reasoning_token_data") if row["parse_success"] else None
        by_pair[row["pair_id"]].append(row)
    accepted = []
    for pair_id, pair in sorted(by_pair.items()):
        if len(pair) == 2 and {row["pair_side"] for row in pair} == {"a", "b"} and all(row["token_data"] for row in pair):
            accepted.extend(sorted(pair, key=lambda row: row["pair_side"]))
    return accepted, len(by_pair) - len(accepted) // 2


def checkpoint_positions(token_data: dict) -> dict[str, int]:
    think_tokens = token_data["thought_token_indices"]
    answer_tokens = token_data["answer_token_indices"]
    first = len(think_tokens) // 3
    second = (2 * len(think_tokens)) // 3
    positions = {
        "early_think": think_tokens[first - 1],
        "middle_think": think_tokens[second - 1],
        "late_think": think_tokens[-1],
        "pre_answer": answer_tokens[0] - 1,
    }
    return positions


def prompt_inputs(adapter, rows: list[dict]):
    prompt = PROMPT_OFFICIAL_SWITCH if adapter.spec.quantized else PROMPT
    return adapter.prepare_inputs(
        [prompt for _ in rows],
        [row["image_path"] for row in rows],
        enable_thinking=True,
    )


def collect_prethink_layers(adapter, rows: list[dict], cache_key: dict, result_dir: Path) -> dict[str, torch.Tensor]:
    path = result_dir / "prethink_layers.pt"
    cached = load_tensor_cache(path, cache_key)
    values = cached.get("values", {})
    pending = [row for row in rows if row["sample_id"] not in values]
    with torch.inference_mode():
        for index, row in enumerate(pending, 1):
            inputs = prompt_inputs(adapter, [row])
            output = adapter.forward(inputs, output_hidden_states=True, use_cache=False)
            position = int(inputs["attention_mask"][0].nonzero()[-1])
            values[row["sample_id"]] = torch.stack([hidden[0, position].half().cpu() for hidden in output.hidden_states])
            if index % CHECKPOINT_EVERY == 0:
                save_tensor_cache(path, cache_key, values=values)
                print(f"[state dynamics] pre-think layers {len(values)}/{len(rows)}", flush=True)
    save_tensor_cache(path, cache_key, values=values)
    return values


def select_layer(prethink: dict[str, torch.Tensor], rows: list[dict], split_ids: dict[str, list[str]]) -> tuple[list[dict], int]:
    by_id = {row["sample_id"]: row for row in rows}
    train_ids, validation_ids = split_ids["train"], split_ids["validation"]
    train_labels = [by_id[sample_id]["larger_target_side"] for sample_id in train_ids]
    validation_labels = [by_id[sample_id]["larger_target_side"] for sample_id in validation_ids]
    layer_count = next(iter(prethink.values())).shape[0]
    output = []
    for layer in range(1, layer_count):
        x_train = torch.stack([prethink[sample_id][layer].float() for sample_id in train_ids])
        x_validation = torch.stack([prethink[sample_id][layer].float() for sample_id in validation_ids])
        predictions = ridge_predictions(x_train, train_labels, x_validation, RIDGE)
        accuracy = label_accuracy(predictions, validation_labels)
        output.append({"layer": layer, "validation_accuracy": round(accuracy, 6)})
    selected = max(output, key=lambda row: (row["validation_accuracy"], -row["layer"]))["layer"]
    return output, int(selected)


def collect_fixed_layer_states(adapter, rows: list[dict], selected_layer: int, cache_key: dict, result_dir: Path) -> dict:
    path = result_dir / "fixed_layer_states.pt"
    cached = load_tensor_cache(path, cache_key, selected_layer=selected_layer)
    features = cached.get("features", {})
    raw_margins = cached.get("raw_margins", {})
    pending = [row for row in rows if row["sample_id"] not in features]
    layer = adapter.decoder_layers()[selected_layer - 1]
    final_norm = adapter.final_norm()
    token_ids = side_token_ids(adapter.tokenizer)

    for index, row in enumerate(pending, 1):
        prompt = prompt_inputs(adapter, [row])
        prompt_len = prompt["input_ids"].shape[1]
        token_data = row["token_data"]
        generation_ids = row["generation_token_ids"][: token_data["answer_token_indices"][-1] + 1]
        gen_ids = torch.tensor([generation_ids], dtype=prompt["input_ids"].dtype)
        gen_mask = torch.ones_like(gen_ids, dtype=prompt["attention_mask"].dtype)
        inputs = adapter.extend_inputs(prompt, gen_ids, gen_mask)
        positions = {"pre_think": prompt_len - 1}
        positions.update({name: prompt_len + position for name, position in checkpoint_positions(token_data).items()})
        captured = {}

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden[0, [positions[name] for name in STATES]].detach()

        handle = layer.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                adapter.forward(inputs, use_cache=False)
        finally:
            handle.remove()
        hidden = captured["hidden"]
        normalized = final_norm(hidden)
        logits = adapter.output_embeddings()(normalized)
        features[row["sample_id"]] = {name: hidden[state_index].float().cpu() for state_index, name in enumerate(STATES)}
        raw_margins[row["sample_id"]] = {
            name: correct_side_margin(logits[state_index], row["larger_target_side"], token_ids)
            for state_index, name in enumerate(STATES)
        }
        if index % CHECKPOINT_EVERY == 0:
            save_tensor_cache(path, cache_key, selected_layer=selected_layer, features=features, raw_margins=raw_margins)
            print(f"[state dynamics] fixed-layer states {len(features)}/{len(rows)}", flush=True)
    save_tensor_cache(path, cache_key, selected_layer=selected_layer, features=features, raw_margins=raw_margins)
    return {"features": features, "raw_margins": raw_margins}


def collect_suffix_margins(adapter, rows: list[dict], cache_key: dict, result_dir: Path) -> dict[str, dict[str, float]]:
    path = result_dir / "answer_suffix_margins.jsonl"
    saved = {row["sample_id"]: row["margins"] for row in read_jsonl(path)} if path.exists() else {}
    unknown = set(saved) - {row["sample_id"] for row in rows}
    if unknown:
        raise SystemExit(f"Suffix checkpoint contains unknown samples: {sorted(unknown)[:3]}")
    token_ids = side_token_ids(adapter.tokenizer)
    suffix_ids = adapter.tokenizer(SUFFIX, add_special_tokens=False).input_ids
    for index, row in enumerate((row for row in rows if row["sample_id"] not in saved), 1):
        prompt = prompt_inputs(adapter, [row] * len(STATES))
        token_data = row["token_data"]
        generation_ids = row["generation_token_ids"]
        positions = checkpoint_positions(token_data)
        prefixes = [[]]
        prefixes.extend(generation_ids[: positions[state] + 1] for state in STATES[1:])
        extensions = [prefix + suffix_ids for prefix in prefixes]
        max_len = max(len(ids) for ids in extensions)
        pad_id = adapter.tokenizer.pad_token_id or adapter.tokenizer.eos_token_id
        gen_ids = torch.full((len(STATES), max_len), pad_id, dtype=prompt["input_ids"].dtype)
        gen_mask = torch.zeros((len(STATES), max_len), dtype=prompt["attention_mask"].dtype)
        for row_index, ids in enumerate(extensions):
            gen_ids[row_index, -len(ids) :] = torch.tensor(ids, dtype=gen_ids.dtype)
            gen_mask[row_index, -len(ids) :] = 1
        inputs = adapter.extend_inputs(prompt, gen_ids, gen_mask)
        with torch.inference_mode():
            logits = adapter.forward(inputs, use_cache=False).logits[:, -1]
        margins = {
            state: correct_side_margin(logits[state_index], row["larger_target_side"], token_ids)
            for state_index, state in enumerate(STATES)
        }
        append_jsonl(path, [{"sample_id": row["sample_id"], "margins": margins}])
        saved[row["sample_id"]] = margins
        if index % 25 == 0:
            print(f"[state dynamics] answer suffix {len(saved)}/{len(rows)}", flush=True)
    return saved


def evaluate(rows: list[dict], split_ids: dict[str, list[str]], test_pairs: list[str], fixed: dict, suffix: dict, selected_layer: int) -> dict:
    by_id = {row["sample_id"]: row for row in rows}
    train_ids, test_ids = split_ids["train"], split_ids["test"]
    train_labels = [by_id[sample_id]["larger_target_side"] for sample_id in train_ids]
    test_labels = [by_id[sample_id]["larger_target_side"] for sample_id in test_ids]
    state_rows = []
    for state in STATES:
        x_train = torch.stack([fixed["features"][sample_id][state] for sample_id in train_ids])
        x_test = torch.stack([fixed["features"][sample_id][state] for sample_id in test_ids])
        predictions = ridge_predictions(x_train, train_labels, x_test, RIDGE)
        probe_values = {sample_id: float(prediction == label) for sample_id, prediction, label in zip(test_ids, predictions, test_labels)}
        raw_values = {sample_id: float(fixed["raw_margins"][sample_id][state]) for sample_id in test_ids}
        suffix_values = {sample_id: float(suffix[sample_id][state]) for sample_id in test_ids}
        state_rows.append(
            {
                "state": state,
                "layer": selected_layer,
                "train_samples": len(train_ids),
                "test_samples": len(test_ids),
                "test_pairs": len(test_pairs),
                **summary_columns("probe_accuracy", probe_values, test_pairs, by_id, accuracy_only=True),
                **summary_columns("raw_usability", raw_values, test_pairs, by_id),
                **summary_columns("answer_suffix_usability", suffix_values, test_pairs, by_id),
            }
        )
    return {"selected_layer": selected_layer, "states": state_rows}


def ridge_predictions(x_train: torch.Tensor, train_labels: list[str], x_test: torch.Tensor, ridge: float) -> list[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = sorted(set(train_labels))
    class_ids = {label: index for index, label in enumerate(classes)}
    x_train, x_test = x_train.float().to(device), x_test.float().to(device)
    mean, std = x_train.mean(0, keepdim=True), x_train.std(0, keepdim=True).clamp_min(1e-6)
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    y = torch.tensor([class_ids[label] for label in train_labels], device=device)
    onehot = torch.nn.functional.one_hot(y, len(classes)).to(x_train.dtype)
    kernel = x_train @ x_train.T + ridge * torch.eye(len(x_train), device=device)
    scores = x_test @ x_train.T @ torch.linalg.solve(kernel, onehot)
    return [classes[index] for index in scores.argmax(1).cpu().tolist()]


def side_token_ids(tokenizer) -> dict[str, list[int]]:
    output = {}
    for side in ("left", "right"):
        ids = {
            int(encoded[0])
            for text in (side, f" {side}")
            if len((encoded := tokenizer(text, add_special_tokens=False).input_ids)) == 1
        }
        if not ids:
            raise SystemExit(f"No single-token candidate for {side}")
        output[side] = sorted(ids)
    return output


def correct_side_margin(logits: torch.Tensor, correct: str, token_ids: dict[str, list[int]]) -> float:
    scores = {side: float(logits[ids].max().item()) for side, ids in token_ids.items()}
    other = "right" if correct == "left" else "left"
    return scores[correct] - scores[other]


def summary_columns(prefix: str, values: dict[str, float], pair_ids: list[str], by_id: dict, accuracy_only: bool = False) -> dict:
    ordered = list(values.values())
    if accuracy_only:
        return {
            prefix: round(mean(ordered), 6),
            f"{prefix}_ci": ci_text(pair_bootstrap(values, pair_ids, by_id, mean)),
        }
    correctness = {sample_id: float(value > 0) for sample_id, value in values.items()}
    return {
        f"{prefix}_margin_mean": round(mean(ordered), 6),
        f"{prefix}_margin_mean_ci": ci_text(pair_bootstrap(values, pair_ids, by_id, mean)),
        f"{prefix}_margin_median": round(median(ordered), 6),
        f"{prefix}_margin_median_ci": ci_text(pair_bootstrap(values, pair_ids, by_id, median)),
        f"{prefix}_accuracy": round(mean(correctness.values()), 6),
        f"{prefix}_accuracy_ci": ci_text(pair_bootstrap(correctness, pair_ids, by_id, mean)),
    }


def pair_bootstrap(values: dict[str, float], pair_ids: list[str], by_id: dict, statistic) -> tuple[float, float]:
    ids_by_pair = defaultdict(list)
    for sample_id in values:
        ids_by_pair[by_id[sample_id]["pair_id"]].append(sample_id)
    if set(ids_by_pair) != set(pair_ids) or any(len(ids) != 2 for ids in ids_by_pair.values()):
        raise SystemExit("Pair bootstrap received an incomplete test pair")
    rng = random.Random(SEED)
    stats = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sampled_pairs = [pair_ids[rng.randrange(len(pair_ids))] for _ in pair_ids]
        sampled_values = [values[sample_id] for pair_id in sampled_pairs for sample_id in ids_by_pair[pair_id]]
        stats.append(statistic(sampled_values))
    stats.sort()
    return stats[int(0.025 * BOOTSTRAP_ROUNDS)], stats[int(0.975 * BOOTSTRAP_ROUNDS)]


def load_tensor_cache(path: Path, cache_key: dict, **expected) -> dict:
    if not path.exists():
        return {}
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("cache_key") != cache_key or any(cache.get(key) != value for key, value in expected.items()):
        raise SystemExit(f"Incompatible answer-state dynamics cache: {path}")
    return cache


def save_tensor_cache(path: Path, cache_key: dict, **payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"cache_key": cache_key, **payload}, temporary)
    temporary.replace(path)


def check_or_write_json(path: Path, expected: dict) -> None:
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != expected:
        raise SystemExit(f"Existing answer-state dynamics provenance does not match: {path}")
    if not path.exists():
        write_json(path, expected)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label_accuracy(predictions: list[str], labels: list[str]) -> float:
    return sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def median(values) -> float:
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def ci_text(interval: tuple[float, float]) -> str:
    return f"[{interval[0]:.6f}, {interval[1]:.6f}]"


if __name__ == "__main__":
    main()
