"""Shared frozen composite behavior implementation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlm_core.io import append_jsonl, read_jsonl, write_json  # noqa: E402
from vlm_core.models import load_vllm_model  # noqa: E402


DATA_DIR = ROOT / "paper/data/occluded_target_reasoning_v2"
RESULT_FAMILY = "occluded_target_reasoning"
PROMPT = (
    "Look at the image. Consider the occluded shapes, their colors, and which side they are on.\n"
    "Which side has more red circles after completing the occluded shapes?\n"
    "A. left\n"
    "B. right\n"
    "Answer only A or B."
)
PROMPT_PROTOCOL = "answer_only_v2"
PARSER_PROTOCOL = "final_channel_ab_v1"
PROTOCOL_VERSION = "2.0.0"
SEED = 13
MAX_NEW_TOKENS = 6144
GPU_MEMORY_UTILIZATION = 0.9
BOOTSTRAP_ROUNDS = 10_000
EXPECTED_DATASET_FILES = {
    "manifest.json": "45ec496d26deb0a5ff1d02195c4dc3cea4cf39b61e40104953e3c51234893232",
    "samples.jsonl": "45c0c74d4360269a35fe30dbd84ca79f5b37c124efc05923c8272918bfaa7f36",
    "pairs.jsonl": "1c1e8ecd3d1e7c18dbd5271c08e071f99f4882994db4eefec73e7915a0116902",
    "splits.jsonl": "25ef342e2f66780a477751cec7b101659f08dec37bfc39594193d02fe0051707",
}
EXPECTED_TRANSFORMERS_VERSION = "5.12.1"
EXPECTED_VLLM_VERSION = "0.25.1"

THINKING_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:
1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.
Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

MODEL_CONFIGS = {
    "qwen": {
        "family": "qwen",
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "result_slug": "qwen3_5_4b",
        "batch_size": 16,
        "sampling": {"temperature": 0.0},
        "reasoning_interface": "chat_template_enable_thinking",
        "reasoning_type": "native",
        "chat_template_fingerprint": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    },
    "gemma": {
        "family": "gemma",
        "model": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "result_slug": "gemma_4_e4b_it",
        "batch_size": 8,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "reasoning_interface": "chat_template_enable_thinking",
        "reasoning_type": "native",
        "chat_template_fingerprint": "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
    },
    "qwen9b": {
        "family": "qwen",
        "model": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "result_slug": "qwen3_5_9b_fp8_dynamic",
        "batch_size": 16,
        "sampling": {"temperature": 0.0},
        "reasoning_interface": "chat_template_enable_thinking",
        "reasoning_type": "native",
        "chat_template_fingerprint": "runtime_official_template",
        "fp8_dynamic": True,
    },
    "gemma12b": {
        "family": "gemma",
        "model": "google/gemma-4-12B-it",
        "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "result_slug": "gemma_4_12b_it_fp8_dynamic",
        "batch_size": 16,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "reasoning_interface": "chat_template_enable_thinking",
        "reasoning_type": "native",
        "chat_template_fingerprint": "runtime_official_template",
        "fp8_dynamic": True,
    },
}


def main(model_key: str) -> None:
    model = model_config(model_key)
    if model.get("load_in_8bit") or model.get("fp8_dynamic"):
        from _model_adapter import generation_adapter

        adapter = generation_adapter(model_key)
    else:
        adapter = None
    if adapter is not None:
        model["chat_template_fingerprint"] = sha256_text(str(getattr(adapter.tokenizer, "chat_template", "")))
        model["runtime_provenance"] = adapter.provenance()
    dataset, samples = load_test_samples()
    image_hashes = validate_images(samples)
    result_dir = ROOT / "paper/results" / model["result_slug"] / RESULT_FAMILY
    result_dir.mkdir(parents=True, exist_ok=True)

    reuse = check_trace_reuse(model_key, dataset, samples)
    manifest = run_manifest(model_key, dataset, samples, reuse, image_hashes=image_hashes)
    check_or_write_json(result_dir / "run_manifest.json", manifest)
    check_or_write_jsonl(result_dir / "sample_manifest.jsonl", sample_manifest(samples, image_hashes))

    full_path = result_dir / "full_thinking_predictions.jsonl"
    no_path = result_dir / "no_thinking_predictions.jsonl"
    if reuse["eligible"]:
        import_reused_predictions(model_key, samples, reuse, manifest["protocol_fingerprint"], full_path)

    existing_full = load_predictions(full_path, samples, "full_thinking", manifest["protocol_fingerprint"])
    existing_no = load_predictions(no_path, samples, "no_thinking", manifest["protocol_fingerprint"])
    validate_prediction_sources(existing_full, "trace_reuse" if reuse["eligible"] else "live_generation")
    validate_prediction_sources(existing_no, "live_generation")
    if len(existing_full) < len(samples) or len(existing_no) < len(samples):
        if model.get("load_in_8bit") or model.get("fp8_dynamic"):
            if len(existing_full) < len(samples):
                generate_condition_8bit(adapter, model_key, model, samples, "full_thinking", full_path, existing_full, manifest)
            if len(existing_no) < len(samples):
                generate_condition_8bit(adapter, model_key, model, samples, "no_thinking", no_path, existing_no, manifest)
        else:
            llm, tokenizer = load_vllm(model)
            if len(existing_full) < len(samples):
                generate_condition(llm, tokenizer, model_key, model, samples, "full_thinking", full_path, existing_full, manifest)
            if len(existing_no) < len(samples):
                generate_condition(llm, tokenizer, model_key, model, samples, "no_thinking", no_path, existing_no, manifest)

    full = load_predictions(full_path, samples, "full_thinking", manifest["protocol_fingerprint"])
    no = load_predictions(no_path, samples, "no_thinking", manifest["protocol_fingerprint"])
    validate_prediction_sources(full, "trace_reuse" if reuse["eligible"] else "live_generation")
    validate_prediction_sources(no, "live_generation")
    if len(full) != len(samples) or len(no) != len(samples):
        raise SystemExit("composite behavior is incomplete; metrics require both conditions on all frozen test samples")
    write_metrics(result_dir, samples, full, no, manifest)


def model_config(model_key: str) -> dict[str, Any]:
    try:
        return MODEL_CONFIGS[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown composite behavior model: {model_key}") from exc


def load_test_samples() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    # Rasterization can differ across systems; keep the construction and split
    # fixed, and record actual file/image hashes in each run's provenance.
    expected = {
        "dataset_id": "occluded_target_reasoning",
        "version": "2.0.0",
        "seed": 260808,
        "sample_count": 1000,
        "pair_count": 500,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise SystemExit("Composite dataset construction settings differ from the frozen protocol")
    if sha256_file(DATA_DIR / "splits.jsonl") != EXPECTED_DATASET_FILES["splits.jsonl"]:
        raise SystemExit("Composite train/test split differs from the frozen protocol")

    split_by_id = {str(row["sample_id"]): str(row["split"]) for row in read_jsonl(DATA_DIR / "splits.jsonl")}
    samples = []
    for dataset_index, row in enumerate(read_jsonl(DATA_DIR / "samples.jsonl")):
        if split_by_id[str(row["sample_id"])] != "test":
            continue
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = DATA_DIR / image_path
        samples.append(
            {
                "sample_id": str(row["sample_id"]),
                "pair_id": str(row["pair_id"]),
                "dataset_index": dataset_index,
                "image_path": str(image_path),
                "question": str(row["question"]),
                "choices": [str(value) for value in row["choices"]],
                "answer": str(row["answer"]),
            }
        )
    validate_test_samples(samples)
    return manifest, samples


def validate_test_samples(samples: list[dict[str, Any]]) -> None:
    if len(samples) != 200 or len({row["sample_id"] for row in samples}) != 200:
        raise SystemExit("composite behavior requires exactly 200 unique canonical test samples")
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        if row["choices"] != ["A", "B"] or row["answer"] not in {"A", "B"}:
            raise SystemExit(f"Invalid choices or answer for {row['sample_id']}")
        by_pair[row["pair_id"]].append(row)
    if len(by_pair) != 100 or any(len(pair) != 2 or {row["answer"] for row in pair} != {"A", "B"} for pair in by_pair.values()):
        raise SystemExit("composite behavior requires 100 complete answer-flipping canonical test pairs")


def validate_images(samples: list[dict[str, Any]]) -> dict[str, str]:
    missing = [row["image_path"] for row in samples if not Path(row["image_path"]).is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} frozen composite behavior images; first: {missing[0]}")
    return {row["sample_id"]: sha256_file(Path(row["image_path"])) for row in samples}


def generation_config(model: dict[str, Any]) -> dict[str, Any]:
    exported = Path(os.environ.get("VLM_CORE_QUANTIZED_ROOT", "quantized_models")) / model["result_slug"] / "compression_manifest.json"
    config = {
        "backend": (
            "transformers_compressed_tensors_fp8_dynamic"
            if model.get("fp8_dynamic") and model["family"] == "gemma"
            else "vllm_compressed_tensors_fp8_dynamic"
            if exported.is_file()
            else "transformers_unsloth"
            if model.get("load_in_8bit")
            else "vllm"
        ),
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed_policy": "seed + original dataset batch start",
        "seed_base": SEED,
        "batch_size": model["batch_size"],
        **model["sampling"],
    }
    if model.get("fp8_dynamic"):
        config.update(weight_precision="fp8", activation_precision="fp8_dynamic")
        if model["family"] == "gemma":
            config["cache_implementation"] = "model_default_gpu"
    elif model.get("load_in_8bit"):
        config.update(load_in_8bit=True, load_in_4bit=False)
    else:
        config.update(
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            transformers=EXPECTED_TRANSFORMERS_VERSION,
            vllm=EXPECTED_VLLM_VERSION,
        )
    return config


def state_dynamics_generation_config(model: dict[str, Any]) -> dict[str, Any]:
    config = generation_config(model)
    config["seed"] = config.pop("seed_base")
    config.pop("seed_policy")
    config.pop("transformers", None)
    config.pop("vllm", None)
    return config


def condition_interface(model_key: str, condition: str) -> dict[str, Any]:
    thinking = condition == "full_thinking"
    return {
        "reasoning_enabled": thinking,
        "reasoning_type": "native" if thinking else "none",
        "chat_template_kwargs": {"enable_thinking": thinking},
        "system_prompt_profile": "model_default",
        "system_prompt_fingerprint": None,
    }


def run_manifest(
    model_key: str,
    dataset: dict[str, Any],
    samples: list[dict[str, Any]],
    reuse: dict[str, Any],
    *,
    image_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    model = model_config(model_key)
    dataset_files = {name: sha256_file(DATA_DIR / name) for name in EXPECTED_DATASET_FILES}
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["version"],
        "dataset_manifest_fingerprint": dataset["fingerprint"],
        "dataset_snapshot_fingerprint": sha256_json(dataset_files),
        "image_set_fingerprint": (
            sha256_json([[row["sample_id"], image_hashes[row["sample_id"]]] for row in samples])
            if image_hashes is not None
            else None
        ),
        "split": "test",
        "sample_ids": [row["sample_id"] for row in samples],
        "pair_ids": sorted({row["pair_id"] for row in samples}),
        "model": model["model"],
        "model_revision": model["revision"],
        "model_runtime": model.get("runtime_provenance"),
        "reasoning_interface": model["reasoning_interface"],
        "reasoning_type": model["reasoning_type"],
        "chat_template_fingerprint": model["chat_template_fingerprint"],
        "prompt": PROMPT,
        "prompt_protocol": PROMPT_PROTOCOL,
        "prompt_fingerprint": sha256_text(PROMPT),
        "dataset_question_fingerprint": sha256_json(sorted({row["question"] for row in samples})),
        "choices_fingerprint": sha256_json([row["choices"] for row in samples]),
        "generation_config": generation_config(model),
        "conditions": {
            name: condition_interface(model_key, name)
            for name in ("full_thinking", "no_thinking")
        },
        "generation_config_fingerprint": sha256_json(generation_config(model)),
        "only_allowed_condition_difference": "reasoning interface (native switch or frozen prompted-reasoning system profile)",
        "parser_protocol": PARSER_PROTOCOL,
        "parse_failure_policy": "incorrect",
        "primary_metrics": ["sample_accuracy", "pair_consistent_accuracy"],
        "bootstrap": {"unit": "pair_id", "rounds": BOOTSTRAP_ROUNDS, "seed": SEED, "interval": "percentile_95"},
    }
    return {
        **protocol,
        "protocol_fingerprint": sha256_json(protocol),
        "dataset_files": dataset_files,
        "image_paths": [portable_path(Path(row["image_path"])) for row in samples],
        "image_sha256_by_sample_id": image_hashes,
        "code_provenance": code_provenance(model_key),
        "outputs": {
            "sample_manifest": "sample_manifest.jsonl",
            "full_thinking": "full_thinking_predictions.jsonl",
            "no_thinking": "no_thinking_predictions.jsonl",
            "metrics": "metrics.json",
            "pair_metrics": "pair_metrics.jsonl",
        },
        "output_schema": {
            "prediction_key": "sample_id",
            "prediction_required_fields": [
                "model",
                "model_revision",
                "sample_id",
                "pair_id",
                "condition",
                "answer",
                "parsed_answer",
                "parse_success",
                "is_correct",
                "final_channel_found",
                "final_text",
                "raw_output",
                "generation_token_ids",
                "generation_tokens",
                "reached_token_limit",
                "source",
                "prompt_fingerprint",
                "generation_config_fingerprint",
                "reasoning_interface_fingerprint",
                "parser_protocol",
                "protocol_fingerprint",
            ],
            "pair_metric_key": "pair_id",
            "metrics_file": "JSON estimates and percentile 95% intervals from pair bootstrap",
        },
        "full_thinking_source": "trace_reuse" if reuse["eligible"] else "live_generation",
        "selective_rerun_policy": "forbidden; resume only the same frozen condition checkpoint",
        "cross_run_merge_policy": "forbidden",
        "trace_reuse": reuse,
    }


def check_trace_reuse(model_key: str, dataset: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    model = model_config(model_key)
    source_dir = ROOT / "paper/results" / model["result_slug"] / "decodability_usability_gap"
    provenance_path = source_dir / "native_thinking_provenance.json"
    trace_path = source_dir / "native_thinking_traces.jsonl"
    reasons = []
    provenance: dict[str, Any] = {}
    traces: list[dict[str, Any]] = []
    if not provenance_path.is_file():
        reasons.append("missing native_thinking_provenance.json")
    else:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not trace_path.is_file():
        reasons.append("missing native_thinking_traces.jsonl")
    else:
        traces = read_jsonl(trace_path)
    if reasons:
        return reuse_report(source_dir, provenance_path, trace_path, reasons, 0)

    expected = {
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["version"],
        "dataset_fingerprint": dataset["fingerprint"],
        "model": model["model"],
        "model_revision": model["revision"],
        "prompt": PROMPT,
        "prompt_fingerprint": sha256_text(PROMPT),
        "generation_config": state_dynamics_generation_config(model),
        "enable_thinking": True,
        "adapter": model_key,
        "precision": "bfloat16",
        "quantization": "none",
        "transformers": EXPECTED_TRANSFORMERS_VERSION,
        "vllm": EXPECTED_VLLM_VERSION,
        "prompt_protocol": "native_thinking_v2",
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            reasons.append(f"provenance mismatch: {key}")

    canonical_ids = [str(row["sample_id"]) for row in read_jsonl(DATA_DIR / "samples.jsonl")]
    if provenance.get("sample_ids") != canonical_ids:
        reasons.append("provenance sample IDs/order do not match the frozen dataset")

    by_id: dict[str, dict[str, Any]] = {}
    for row in traces:
        sample_id = str(row.get("sample_id", ""))
        if sample_id in by_id:
            reasons.append(f"duplicate trace sample: {sample_id}")
            break
        by_id[sample_id] = row
    if [str(row.get("sample_id", "")) for row in traces] != canonical_ids:
        reasons.append("trace sample IDs/order do not match the frozen dataset")
    covered = 0
    for sample in samples:
        row = by_id.get(sample["sample_id"])
        if row is None:
            reasons.append(f"missing test trace: {sample['sample_id']}")
            continue
        covered += 1
        if row.get("pair_id") != sample["pair_id"] or row.get("answer") != sample["answer"] or row.get("split") != "test":
            reasons.append(f"trace/sample mismatch: {sample['sample_id']}")
        if not isinstance(row.get("full_generation"), str) or not row["full_generation"]:
            reasons.append(f"missing full generation: {sample['sample_id']}")
        if not isinstance(row.get("generation_token_ids"), list) or not row["generation_token_ids"]:
            reasons.append(f"missing generation token IDs: {sample['sample_id']}")
    return reuse_report(source_dir, provenance_path, trace_path, sorted(set(reasons)), covered)


def reuse_report(source_dir: Path, provenance_path: Path, trace_path: Path, reasons: list[str], covered: int) -> dict[str, Any]:
    return {
        "eligible": not reasons,
        "source_dir": portable_path(source_dir),
        "provenance_path": portable_path(provenance_path),
        "trace_path": portable_path(trace_path),
        "covered_test_samples": covered,
        "required_test_samples": 200,
        "reasons": reasons,
        "provenance_sha256": sha256_file(provenance_path) if provenance_path.is_file() else None,
        "trace_sha256": sha256_file(trace_path) if trace_path.is_file() else None,
    }


def import_reused_predictions(
    model_key: str,
    samples: list[dict[str, Any]],
    reuse: dict[str, Any],
    protocol_fingerprint: str,
    output_path: Path,
) -> None:
    existing = load_predictions(output_path, samples, "full_thinking", protocol_fingerprint)
    if existing:
        if len(existing) != len(samples) or {row.get("source") for row in existing.values()} != {"trace_reuse"}:
            raise SystemExit("Partial or mixed-source composite behavior full-thinking output cannot be merged with answer-state dynamics")
        return
    source = {str(row["sample_id"]): row for row in read_jsonl(ROOT / reuse["trace_path"])}
    rows = []
    for sample in samples:
        trace = source[sample["sample_id"]]
        rows.append(
            prediction_row(
                model_key,
                sample,
                "full_thinking",
                str(trace["full_generation"]),
                [int(value) for value in trace["generation_token_ids"]],
                protocol_fingerprint,
                "trace_reuse",
            )
        )
    append_jsonl(output_path, rows)


def load_vllm(model: dict[str, Any]):
    versions = {"transformers": package_version("transformers"), "vllm": package_version("vllm")}
    expected = {"transformers": EXPECTED_TRANSFORMERS_VERSION, "vllm": EXPECTED_VLLM_VERSION}
    if versions != expected:
        raise SystemExit(f"composite behavior requires the locked model environment: {versions} != {expected}")

    kwargs = {
        "revision": model["revision"],
        "dtype": "bfloat16",
        "max_model_len": MAX_NEW_TOKENS + 2048,
        "max_num_seqs": model["batch_size"],
        "limit_mm_per_prompt": {"image": 1},
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    }
    if model["reasoning_interface"] == "thinking_system_prompt":
        kwargs["trust_remote_code"] = True
    llm = load_vllm_model(model["model"], **kwargs)
    tokenizer = llm.get_tokenizer()
    actual_template = sha256_text(str(getattr(tokenizer, "chat_template", "")))
    if actual_template != model["chat_template_fingerprint"]:
        raise SystemExit(f"Chat-template fingerprint mismatch: {actual_template}")
    return llm, tokenizer


def generate_condition(llm, tokenizer, model_key, model, samples, condition, output_path, existing, manifest) -> None:
    from PIL import Image
    from vllm import SamplingParams

    by_start: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if sample["sample_id"] not in existing:
            start = (sample["dataset_index"] // model["batch_size"]) * model["batch_size"]
            by_start[start].append(sample)
    for start, batch in sorted(by_start.items()):
        images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
        sampling_kwargs = {
            key: value for key, value in model["sampling"].items() if key != "do_sample"
        }
        sampling_kwargs.update(max_tokens=MAX_NEW_TOKENS, seed=SEED + start)
        sampling = SamplingParams(**sampling_kwargs)
        try:
            outputs = llm.chat(
                [vllm_message(model_key, condition, image) for image in images],
                sampling,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": condition == "full_thinking"},
            )
        finally:
            for image in images:
                image.close()
        if len(outputs) != len(batch) or any(len(output.outputs) != 1 for output in outputs):
            raise SystemExit("vLLM did not return exactly one generation per composite behavior sample")
        rows = []
        for sample, output in zip(batch, outputs):
            token_ids = [int(value) for value in output.outputs[0].token_ids]
            raw = tokenizer.decode(token_ids, skip_special_tokens=False)
            rows.append(
                prediction_row(
                    model_key,
                    sample,
                    condition,
                    raw,
                    token_ids,
                    manifest["protocol_fingerprint"],
                    "live_generation",
                )
            )
        append_jsonl(output_path, rows)
        existing.update((row["sample_id"], row) for row in rows)
        print(f"[composite/{model_key}/{condition}] {len(existing)}/{len(samples)}", flush=True)


def generate_condition_8bit(adapter, model_key, model, samples, condition, output_path, existing, manifest) -> None:
    by_start: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample_index, sample in enumerate(samples):
        if sample["sample_id"] not in existing:
            start = (sample_index // model["batch_size"]) * model["batch_size"]
            by_start[start].append(sample)
    for start, batch in sorted(by_start.items()):
        outputs = adapter.generate(
            [PROMPT for _ in batch],
            [row["image_path"] for row in batch],
            enable_thinking=condition == "full_thinking",
            max_new_tokens=MAX_NEW_TOKENS,
            seed=SEED + start,
        )
        rows = [
            prediction_row(
                model_key,
                sample,
                condition,
                output["text"],
                output["token_ids"],
                manifest["protocol_fingerprint"],
                "live_generation",
            )
            for sample, output in zip(batch, outputs)
        ]
        append_jsonl(output_path, rows)
        existing.update((row["sample_id"], row) for row in rows)
        print(f"[composite/{model_key}/{condition}] {len(existing)}/{len(samples)}", flush=True)


def vllm_message(model_key: str, condition: str, image) -> list[dict[str, Any]]:
    user = {
        "role": "user",
        "content": [{"type": "image_pil", "image_pil": image}, {"type": "text", "text": PROMPT}],
    }
    return [user]




def prediction_row(model_key, sample, condition, raw, token_ids, protocol_fingerprint, source) -> dict[str, Any]:
    model = model_config(model_key)
    final_text, final_channel_found = final_channel(model_key, condition, raw)
    parsed = parse_choice(final_text) if final_channel_found else None
    return {
        "model": model["model"],
        "model_revision": model["revision"],
        "sample_id": sample["sample_id"],
        "pair_id": sample["pair_id"],
        "condition": condition,
        "answer": sample["answer"],
        "parsed_answer": parsed or "",
        "parse_success": parsed is not None,
        "is_correct": parsed == sample["answer"],
        "final_channel_found": final_channel_found,
        "final_text": final_text,
        "raw_output": raw,
        "generation_token_ids": token_ids,
        "generation_tokens": len(token_ids),
        "reached_token_limit": len(token_ids) >= MAX_NEW_TOKENS,
        "source": source,
        "prompt_fingerprint": sha256_text(PROMPT),
        "generation_config_fingerprint": sha256_json(generation_config(model)),
        "reasoning_interface_fingerprint": sha256_json(condition_interface(model_key, condition)),
        "parser_protocol": PARSER_PROTOCOL,
        "protocol_fingerprint": protocol_fingerprint,
    }


def final_channel(model_key: str, condition: str, text: str) -> tuple[str, bool]:
    if condition == "no_thinking":
        return text.strip(), True
    if model_config(model_key)["family"] == "gemma":
        match = re.search(r"<\|channel>thought\n.*?<channel\|>", text, flags=re.DOTALL)
        if not match:
            return "", False
        final = re.sub(r"^\s*<\|channel>final\n", "", text[match.end() :]).strip()
        return final, bool(final)
    end = list(re.finditer(r"</think>", text, flags=re.IGNORECASE))
    if not end:
        return "", False
    final = text[end[-1].end() :].strip()
    return final, bool(final)


def parse_choice(text: str) -> str | None:
    candidates: list[tuple[int, str]] = []
    patterns = (
        r"\\boxed\{\s*([AB])\s*\}",
        r"(?:final answer|answer|option|choice)\s*(?:is|:|should be|would be)?\s*\**\s*([AB])\b",
        r"\b([AB])\s*\.\s*(?:left|right)\b",
        r"(?:^|\n)\s*(?:\*\*)?([AB])(?:\*\*)?\s*(?=$|\n|<)",
    )
    for pattern in patterns:
        candidates.extend((match.start(1), match.group(1).upper()) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return max(candidates)[1] if candidates else None


def load_predictions(path: Path, samples: list[dict[str, Any]], condition: str, protocol_fingerprint: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    expected = {row["sample_id"]: row for row in samples}
    output = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in expected or sample_id in output:
            raise SystemExit(f"Invalid or duplicate composite behavior prediction: {sample_id}")
        sample = expected[sample_id]
        model_key = model_config_for_prediction(row)
        final_text, final_channel_found = final_channel(model_key, condition, str(row.get("raw_output", "")))
        parsed = parse_choice(final_text) if final_channel_found else None
        token_ids = row.get("generation_token_ids")
        if (
            row.get("condition") != condition
            or row.get("model_revision") != model_config(model_key)["revision"]
            or row.get("pair_id") != sample["pair_id"]
            or row.get("answer") != sample["answer"]
            or row.get("protocol_fingerprint") != protocol_fingerprint
            or row.get("prompt_fingerprint") != sha256_text(PROMPT)
            or row.get("generation_config_fingerprint") != sha256_json(generation_config(model_config(model_key)))
            or row.get("reasoning_interface_fingerprint") != sha256_json(condition_interface(model_key, condition))
            or row.get("parser_protocol") != PARSER_PROTOCOL
            or not isinstance(row.get("raw_output"), str)
            or not isinstance(token_ids, list)
            or any(not isinstance(value, int) for value in token_ids)
            or row.get("generation_tokens") != len(token_ids)
            or row.get("reached_token_limit") != (len(token_ids) >= MAX_NEW_TOKENS)
            or row.get("final_channel_found") != final_channel_found
            or row.get("final_text") != final_text
            or row.get("parsed_answer") != (parsed or "")
            or row.get("parse_success") != (parsed is not None)
            or row.get("is_correct") != (parsed == sample["answer"])
        ):
            raise SystemExit(f"Incompatible composite behavior prediction checkpoint: {sample_id}")
        output[sample_id] = row
    return output


def validate_prediction_sources(rows: dict[str, dict[str, Any]], expected: str) -> None:
    if rows and {row.get("source") for row in rows.values()} != {expected}:
        raise SystemExit(f"composite behavior predictions cannot merge sources; expected only {expected}")


def write_metrics(result_dir: Path, samples, full_by_id, no_by_id, manifest) -> None:
    full = [full_by_id[row["sample_id"]] for row in samples]
    no = [no_by_id[row["sample_id"]] for row in samples]
    pair_rows = paired_rows(samples, full_by_id, no_by_id)
    metrics = metric_summary(pair_rows)
    metrics.update(
        {
            "model": manifest["model"],
            "model_revision": manifest["model_revision"],
            "protocol_fingerprint": manifest["protocol_fingerprint"],
            "sample_accuracy_minus_pair_consistent": {
                "full_thinking": metrics["metrics"]["full_thinking_sample_minus_pair_consistent"],
                "no_thinking": metrics["metrics"]["no_thinking_sample_minus_pair_consistent"],
            },
            "expected_samples": len(samples),
            "expected_pairs": len(pair_rows),
            "conditions": {
                "full_thinking": condition_counts(full),
                "no_thinking": condition_counts(no),
            },
        }
    )
    write_json(result_dir / "metrics.json", metrics)
    pair_path = result_dir / "pair_metrics.jsonl"
    if pair_path.exists():
        existing = read_jsonl(pair_path)
        if existing != pair_rows:
            raise SystemExit("Existing composite behavior pair metrics do not match predictions")
    else:
        append_jsonl(pair_path, pair_rows)
    write_json(
        result_dir / "run_summary.json",
        {
            "status": "formal_complete",
            "sample_count": len(samples),
            "pair_count": len(pair_rows),
            "full_thinking_source": manifest["full_thinking_source"],
            "metrics_path": portable_path(result_dir / "metrics.json"),
        },
    )
    write_runtime_provenance(result_dir)


def paired_rows(samples, full_by_id, no_by_id) -> list[dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_pair[sample["pair_id"]].append(sample)
    rows = []
    for pair_id, pair in sorted(by_pair.items()):
        ids = [row["sample_id"] for row in pair]
        full_correct = [bool(full_by_id[sample_id]["is_correct"]) for sample_id in ids]
        no_correct = [bool(no_by_id[sample_id]["is_correct"]) for sample_id in ids]
        rows.append(
            {
                "pair_id": pair_id,
                "sample_ids": ids,
                "full_thinking_sample_correct": full_correct,
                "no_thinking_sample_correct": no_correct,
                "full_thinking_pair_consistent": all(full_correct),
                "no_thinking_pair_consistent": all(no_correct),
            }
        )
    return rows


def condition_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "parse_success": sum(bool(row["parse_success"]) for row in rows),
        "parse_rate": round(sum(bool(row["parse_success"]) for row in rows) / len(rows), 6),
        "correct": sum(bool(row["is_correct"]) for row in rows),
        "accuracy": round(sum(bool(row["is_correct"]) for row in rows) / len(rows), 6),
        "token_limit_reached": sum(bool(row["reached_token_limit"]) for row in rows),
    }


def metric_summary(pair_rows: list[dict[str, Any]], rounds: int = BOOTSTRAP_ROUNDS, seed: int = SEED) -> dict[str, Any]:
    estimates = statistics_for(pair_rows)
    rng = random.Random(seed)
    boot = {name: [] for name in estimates}
    for _ in range(rounds):
        sampled = [pair_rows[rng.randrange(len(pair_rows))] for _ in pair_rows]
        values = statistics_for(sampled)
        for name, value in values.items():
            boot[name].append(value)
    return {
        "bootstrap_unit": "pair_id",
        "bootstrap_rounds": rounds,
        "bootstrap_seed": seed,
        "metrics": {
            name: {"estimate": round(value, 6), "ci95": [round(percentile(values, 0.025), 6), round(percentile(values, 0.975), 6)]}
            for name, value, values in ((name, estimate, boot[name]) for name, estimate in estimates.items())
        },
    }


def statistics_for(pair_rows: list[dict[str, Any]]) -> dict[str, float]:
    full_samples = [value for row in pair_rows for value in row["full_thinking_sample_correct"]]
    no_samples = [value for row in pair_rows for value in row["no_thinking_sample_correct"]]
    full_pairs = [bool(row["full_thinking_pair_consistent"]) for row in pair_rows]
    no_pairs = [bool(row["no_thinking_pair_consistent"]) for row in pair_rows]
    sample_full = sum(full_samples) / len(full_samples)
    sample_no = sum(no_samples) / len(no_samples)
    pair_full = sum(full_pairs) / len(full_pairs)
    pair_no = sum(no_pairs) / len(no_pairs)
    return {
        "full_thinking_sample_accuracy": sample_full,
        "no_thinking_sample_accuracy": sample_no,
        "sample_accuracy_difference": sample_full - sample_no,
        "full_thinking_pair_consistent_accuracy": pair_full,
        "no_thinking_pair_consistent_accuracy": pair_no,
        "pair_consistent_accuracy_difference": pair_full - pair_no,
        "full_thinking_sample_minus_pair_consistent": sample_full - pair_full,
        "no_thinking_sample_minus_pair_consistent": sample_no - pair_no,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def write_runtime_provenance(result_dir: Path) -> None:
    write_json(
        result_dir / "runtime_provenance.json",
        {
            "python": sys.version,
            "vllm": package_version("vllm"),
            "transformers": package_version("transformers"),
            "unsloth": package_version("unsloth"),
            "bitsandbytes": package_version("bitsandbytes"),
        },
    )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_or_write_json(path: Path, expected: dict[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise SystemExit(f"Existing composite behavior manifest is incompatible: {path}")
    else:
        write_json(path, expected)


def check_or_write_jsonl(path: Path, expected: list[dict[str, Any]]) -> None:
    if path.is_file():
        if read_jsonl(path) != expected:
            raise SystemExit(f"Existing composite behavior sample manifest is incompatible: {path}")
    else:
        append_jsonl(path, expected)


def sample_manifest(samples: list[dict[str, Any]], image_hashes: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "image_path": portable_path(Path(row["image_path"])),
            "image_sha256": image_hashes[row["sample_id"]],
            "inference_prompt": PROMPT,
            "inference_prompt_fingerprint": sha256_text(PROMPT),
        }
        for row in samples
    ]


def model_config_for_prediction(row: dict[str, Any]) -> str:
    model_name = row.get("model")
    for key, config in MODEL_CONFIGS.items():
        if config["model"] == model_name:
            return key
    raise SystemExit(f"Unknown model in composite behavior prediction: {model_name}")


def code_provenance(model_key: str) -> dict[str, Any]:
    wrapper = ROOT / "paper/experiments" / f"occluded_target_reasoning_{model_config(model_key)['result_slug']}.py"
    state = repository_state()
    return {
        **state,
        "shared_entrypoint": portable_path(Path(__file__).resolve()),
        "shared_entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "model_entrypoint": portable_path(wrapper),
        "model_entrypoint_sha256": sha256_file(wrapper),
        "model_loader": portable_path(ROOT / "src/vlm_core/models.py"),
        "model_loader_sha256": sha256_file(ROOT / "src/vlm_core/models.py"),
    }


def repository_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    return {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
    }


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dry_run() -> dict[str, Any]:
    dataset, samples = load_test_samples()
    models = {}
    for model_key in MODEL_CONFIGS:
        reuse = check_trace_reuse(model_key, dataset, samples)
        manifest = run_manifest(model_key, dataset, samples, reuse)
        models[model_key] = {
            "model": manifest["model"],
            "result_dir": str(ROOT / "paper/results" / MODEL_CONFIGS[model_key]["result_slug"] / RESULT_FAMILY),
            "preflight_protocol_fingerprint": manifest["protocol_fingerprint"],
            "trace_reuse": reuse,
            "missing_images": sum(not Path(row["image_path"]).is_file() for row in samples),
        }
    return {
        "status": "dry_run_passed",
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["version"],
        "dataset_fingerprint": dataset["fingerprint"],
        "test_samples": len(samples),
        "test_pairs": len({row["pair_id"] for row in samples}),
        "prompt_fingerprint": sha256_text(PROMPT),
        "models": models,
    }
