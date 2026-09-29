"""Run formal two-fold external early stopping on the full MMStar benchmark."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from _model_adapter import (
    ModelAdapter,
    VllmGenerationAdapter,
    generation_adapter,
    model_spec,
    reasoning_span,
)
from torch import nn

from vlm_core.models import freeze_model
from vlm_core.schema import image_path_from_messages
from vlm_core.benchmark_answers import (
    SCORING_VERSION as BENCHMARK_ANSWER_SCORING_VERSION,
    choice_answer,
)
from vlm_core.io import append_jsonl, read_jsonl, write_json, write_jsonl
from vlm_core.realworldqa_answers import (
    SCORING_VERSION as REALWORLDQA_ANSWER_SCORING_VERSION,
    extract_choice_answer as extract_realworldqa_choice,
    extract_direct_answer,
    extract_final_answer as extract_realworldqa_final,
)
from vlm_core.thinking_state import decoder_layers

# Frozen experiment settings.
ROOT = Path(__file__).resolve().parents[2]
FORMAL_MODELS = {
    "qwen": {
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "result_slug": "qwen3_5_4b",
        "family": "qwen",
        "batch_size": 16,
        "layer": 17,
    },
    "qwen9b": {
        "model": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "result_slug": "qwen3_5_9b_fp8_dynamic",
        "family": "qwen",
        "batch_size": 16,
        "layer": 16,
    },
    "gemma": {
        "model": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "result_slug": "gemma_4_e4b_it",
        "family": "gemma",
        "batch_size": 8,
        "layer": 1,
    },
    "gemma12b": {
        "model": "google/gemma-4-12B-it",
        "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "result_slug": "gemma_4_12b_it_fp8_dynamic",
        "family": "gemma",
        "batch_size": 4,
        "layer": 20,
    },
}
MODEL_ENV = "VLM_CORE_MMSTAR_EARLY_STOP_MODEL"
MODEL_KEY = os.environ.get(MODEL_ENV, "qwen")
if MODEL_KEY not in FORMAL_MODELS:
    raise ValueError(f"Unknown formal MMStar model key: {MODEL_KEY}")
MODEL_CONFIG = FORMAL_MODELS[MODEL_KEY]
MODEL_SPEC = model_spec(MODEL_KEY)
MODEL = MODEL_CONFIG["model"]
MODEL_REVISION = MODEL_CONFIG["revision"]
LAYER = MODEL_CONFIG["layer"]
CHECKPOINT_INTERVAL = 256
MAX_NEW_TOKENS = 6144
ANSWER_MAX_TOKENS = 128
SAVE_GENERATED_TOKEN_IDS = os.environ.get("VLM_CORE_SAVE_GENERATED_TOKEN_IDS") == "1"
MODEL_CONTEXT_LENGTH = MAX_NEW_TOKENS + 2048
ANSWER_CUE = "Therefore, the answer is"
INITIAL_BATCH_SIZE = MODEL_CONFIG["batch_size"]
ADAPTIVE_BATCH_SIZE = 4 if MODEL_KEY == "qwen9b" else INITIAL_BATCH_SIZE
TRAIN_FRACTION = 0.8
TRAIN_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 73
EXPECTED_SAMPLES = 1500
EXPECTED_FOLD_SIZE = 750
FOLD_SIZES = {"A": 750, "B": 750}
MAX_ACCURACY_DROP = 0.005
MANIFEST = ROOT / "paper/data/mmstar/manifest.jsonl"
SOURCE_TRACES = {
    "A": ROOT / "paper/data/mmstar_external_early_stopping" / MODEL_CONFIG["result_slug"] / "fold_a_native_thinking.jsonl",
    "B": ROOT / "paper/data/mmstar_external_early_stopping" / MODEL_CONFIG["result_slug"] / "fold_b_native_thinking.jsonl",
}
DATA_DIR = ROOT / "paper/data/mmstar_external_early_stopping" / MODEL_CONFIG["result_slug"]
RESULT_DIR = ROOT / "paper/results" / MODEL_CONFIG["result_slug"] / "mmstar_external_early_stopping"
STAGE_ENV = "VLM_CORE_MMSTAR_USABILITY_STOP_STAGE"
PROTOCOL = "mmstar_external_early_stopping"
DATASET_ENV = "VLM_CORE_EARLY_STOP_DATASET"
DATASET = os.environ.get(DATASET_ENV, "mmstar")
if DATASET not in {"mmstar", "realworldqa"}:
    raise ValueError(f"Unknown early-stopping dataset: {DATASET}")
if DATASET == "realworldqa":
    if MODEL_KEY not in {"qwen", "qwen9b", "gemma", "gemma12b"}:
        raise ValueError("RealWorldQA is frozen to the four formal Qwen/Gemma models.")
    EXPECTED_SAMPLES = 765
    FOLD_SIZES = {"A": 383, "B": 382}
    if MODEL_KEY in {"qwen", "gemma"}:
        INITIAL_BATCH_SIZE = ADAPTIVE_BATCH_SIZE = 16
    MANIFEST = ROOT / "paper/data/realworldqa/manifest.jsonl"
    DATA_DIR = ROOT / "paper/data/realworldqa_external_early_stopping" / MODEL_CONFIG["result_slug"]
    RESULT_DIR = ROOT / "paper/results" / MODEL_CONFIG["result_slug"] / "realworldqa_external_early_stopping"
    SOURCE_TRACES = {fold: DATA_DIR / f"fold_{fold.lower()}_native_thinking.jsonl" for fold in ("A", "B")}
    PROTOCOL = "realworldqa_external_early_stopping_v1"
    split_root = ROOT / "paper/data/realworldqa_external_early_stopping"
    FOLD_BY_CASE = {}
    for fold in ("A", "B"):
        members = read_jsonl(split_root / f"fold_{fold.lower()}.jsonl")
        if len(members) != FOLD_SIZES[fold]:
            raise ValueError(f"Wrong frozen fold {fold} size.")
        for row in members:
            case = int(row["source_row_index"])
            if case in FOLD_BY_CASE or row["fold"] != fold:
                raise ValueError("Frozen folds contain duplicate IDs or wrong fold labels.")
            FOLD_BY_CASE[case] = fold
    SPLIT_METADATA = json.loads((split_root / "split.json").read_text(encoding="utf-8"))
    if (set(FOLD_BY_CASE) != set(range(EXPECTED_SAMPLES))
            or SPLIT_METADATA["dataset_id"] != "xai-org/RealworldQA"
            or SPLIT_METADATA["dataset_revision"] != "17e7f75e092e47169732462ea3cdfebe911105dd"
            or SPLIT_METADATA["seed"] != SEED):
        raise ValueError("RealWorldQA split source, seed, or full coverage mismatch.")
    for filename, expected_hash in SPLIT_METADATA["sha256"].items():
        if hashlib.sha256((split_root / filename).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Frozen fold checksum mismatch: {filename}")
    IMAGE_GROUPS = SPLIT_METADATA["image_groups"]
    for group in IMAGE_GROUPS:
        if len({FOLD_BY_CASE[case] for case in group}) != 1:
            raise ValueError("Duplicate image crosses outer folds.")
ANSWER_SCORING_VERSION = (
    REALWORLDQA_ANSWER_SCORING_VERSION
    if DATASET == "realworldqa"
    else BENCHMARK_ANSWER_SCORING_VERSION
)
# The scorer supplies both the forced-stop labels and the validation metrics
# that threshold selection compares against, so a detector may only be reused
# once it carries the current answer-scoring revision.
REQUIRE_ANSWER_SCORING_REFRESH = DATASET == "mmstar" or (
    DATASET == "realworldqa" and MODEL_KEY != "qwen"
)
RESULT_DIR_ENV = "VLM_CORE_EARLY_STOP_RESULT_DIR"
if os.environ.get(RESULT_DIR_ENV):
    RESULT_DIR = ROOT / os.environ[RESULT_DIR_ENV]

STAGES = (
    "prepare",
    "trace_A",
    "trace_B",
    "cache_A",
    "train_A",
    "evaluate_A_to_B",
    "cache_B",
    "train_B",
    "evaluate_B_to_A",
    "summarize",
)


def main() -> None:
    stage = os.environ.get(STAGE_ENV)
    if stage:
        run_stage(stage)
        return
    for name in STAGES:
        log(f"starting isolated stage {name}")
        environment = os.environ.copy()
        environment[STAGE_ENV] = name
        subprocess.run([sys.executable, str(Path(__file__).resolve())], check=True, env=environment)


def run_stage(stage: str) -> None:
    seed_all(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    actions = {
        "prepare": prepare,
        "trace_A": lambda: prepare_native_traces("A"),
        "trace_B": lambda: prepare_native_traces("B"),
        "cache_A": lambda: cache_fold("A"),
        "train_A": lambda: train_direction("A", "B"),
        "evaluate_A_to_B": lambda: evaluate_direction_adaptive_only("A", "B"),
        "cache_B": lambda: cache_fold("B"),
        "train_B": lambda: train_direction("B", "A"),
        "evaluate_B_to_A": lambda: evaluate_direction_adaptive_only("B", "A"),
        "summarize": summarize_external,
    }
    if stage not in actions:
        raise SystemExit(f"Unknown stage {stage!r}.")
    actions[stage]()


def log(message: str) -> None:
    print(f"[{DATASET}-usability-stop] {message}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rows() -> list[dict[str, Any]]:
    values = read_jsonl(MANIFEST)
    ids = sorted(int(row["index"]) for row in values)
    if len(values) != EXPECTED_SAMPLES or ids != list(range(EXPECTED_SAMPLES)):
        raise SystemExit(f"{DATASET} manifest must contain stable case IDs 0..{EXPECTED_SAMPLES - 1} exactly once.")
    return values


def fold_for(case_id: int) -> str:
    return FOLD_BY_CASE[case_id] if DATASET == "realworldqa" else ("A" if case_id % 2 == 0 else "B")


def fold_rows(fold: str) -> list[dict[str, Any]]:
    selected = [row for row in rows() if fold_for(int(row["index"])) == fold]
    if len(selected) != FOLD_SIZES[fold]:
        raise SystemExit(f"Fold {fold} has {len(selected)} rows, expected {FOLD_SIZES[fold]}.")
    return selected


def latest_by_case(path: Path, *, require_protocol: bool = True) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    values = read_jsonl(path)
    if require_protocol and any(
        row.get("protocol") != PROTOCOL
        or row.get("model") != MODEL
        or row.get("model_revision") != MODEL_REVISION
        for row in values
    ):
        raise SystemExit(f"Refusing incompatible checkpoint rows in {path}.")
    return {int(row["case_id"]): row for row in values}


def latest_prediction_by_case(path: Path) -> dict[int, dict[str, Any]]:
    values = latest_by_case(path)
    if REQUIRE_ANSWER_SCORING_REFRESH:
        values = {
            case: row for case, row in values.items()
            if row.get("answer_scoring_version") == ANSWER_SCORING_VERSION
        }
    return values


def trace_rows(fold: str) -> dict[int, dict[str, Any]]:
    path = SOURCE_TRACES[fold]
    traces = latest_by_case(path)
    expected = {int(row["index"]) for row in fold_rows(fold)}
    present = {case: row for case, row in traces.items() if case in expected and row.get("token_ids")}
    if len(present) != FOLD_SIZES[fold]:
        raise SystemExit(
            f"Fold {fold} has {len(present)}/{FOLD_SIZES[fold]} native trajectories; "
            f"run trace_{fold} first."
        )
    if DATASET == "realworldqa":
        expected_hash = json.loads((RESULT_DIR / "run_config.json").read_text(encoding="utf-8"))["chat_template_sha256"]
        if any(row.get("generation_config", {}).get("chat_template_sha256") != expected_hash
               for row in present.values()):
            raise SystemExit("Native trace template fingerprint differs from frozen run config.")
    return present


def prepare_native_traces(fold: str) -> None:
    generate_native_traces(fold)
    trace_rows(fold)


def generate_native_traces(fold: str) -> None:
    path = SOURCE_TRACES[fold]
    existing = latest_by_case(path)
    pending = [row for row in fold_rows(fold) if int(row["index"]) not in existing]
    if not pending:
        return

    bundle = native_generation_adapter()
    generation_config = bundle.generation_provenance(True, MAX_NEW_TOKENS, SEED)
    if DATASET == "realworldqa":
        generation_config["chat_template_sha256"] = verify_chat_template(bundle.tokenizer)
    batch_size = INITIAL_BATCH_SIZE
    offset = 0
    while offset < len(pending):
        batch = pending[offset : offset + batch_size]
        try:
            outputs = bundle.generate(
                [question_text(row) for row in batch],
                [image_path_from_messages(row["messages"]) for row in batch],
                enable_thinking=True,
                max_new_tokens=MAX_NEW_TOKENS,
                seed=SEED,
            )
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
            log(f"fold {fold} native traces: OOM, reducing batch size to {batch_size}")
            continue
        saved = []
        for row, output in zip(batch, outputs):
            token_ids = [int(value) for value in output["token_ids"]]
            trace = parsed_trace(bundle.tokenizer, token_ids)
            generation = trace["canonical_generation"]
            saved.append({
                "case_id": int(row["index"]),
                "fold": fold,
                "gold": str(row["answer"]).upper(),
                "token_ids": token_ids,
                "reasoning_token_indices": trace["reasoning_token_indices"],
                "final_start": trace["final_start"],
                "generation": generation,
                "prediction": extract_row_choice(generation, row, require_closed_thinking=True),
                "closed_thinking": trace["closed_thinking"],
                "generation_tokens": len(token_ids),
                "batch_size": len(batch),
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "max_new_tokens": MAX_NEW_TOKENS,
                "seed": SEED,
                "generation_config": generation_config,
                "protocol": PROTOCOL,
            })
        append_jsonl(path, saved)
        offset += len(batch)
        log(f"fold {fold} native traces {offset}/{len(pending)} batch={batch_size}")
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def native_generation_adapter():
    if MODEL_SPEC.fp8_dynamic:
        return generation_adapter(
            MODEL_KEY,
            context_length=MODEL_CONTEXT_LENGTH,
            max_running_requests=INITIAL_BATCH_SIZE,
        )
    return VllmGenerationAdapter.load_bfloat16(
        MODEL_KEY, max_model_len=MODEL_CONTEXT_LENGTH,
        max_num_seqs=INITIAL_BATCH_SIZE if DATASET == "realworldqa" else None,
    )


def question_text(row: dict[str, Any]) -> str:
    return str(row.get("question") or next(
        item["value"] for item in row["messages"] if item.get("type") == "text"
    ))


def parsed_trace(tokenizer, token_ids: list[int]) -> dict[str, Any]:
    span = reasoning_span(
        MODEL_SPEC, tokenizer, token_ids,
        min_thought_tokens=0 if DATASET == "realworldqa" else 6,
    )
    if span is None:
        return {
            "closed_thinking": False,
            "reasoning_token_indices": list(range(len(token_ids))),
            "final_start": None,
            "canonical_generation": tokenizer.decode(token_ids, skip_special_tokens=False).strip(),
        }
    final_start = int(span["final_start"])
    final = tokenizer.decode(token_ids[final_start:], skip_special_tokens=False).strip()
    return {
        "closed_thinking": True,
        "reasoning_token_indices": [int(index) for index in span["thought_token_indices"]],
        "final_start": final_start,
        "canonical_generation": f"</think>\n{final}",
    }


def trace_reasoning_tokens(trace: dict[str, Any]) -> int:
    return len(trace["reasoning_token_indices"])


def split_ids(fold: str) -> tuple[list[int], list[int]]:
    ids = [int(row["index"]) for row in fold_rows(fold)]
    boundary = int(len(ids) * TRAIN_FRACTION)
    if DATASET == "realworldqa":
        groups_by_case = {case: group for group in IMAGE_GROUPS for case in group}
        groups = [groups_by_case.get(case, [case]) for case in sorted(ids)
                  if case == min(groups_by_case.get(case, [case]))]
        random.Random(SEED + ord(fold)).shuffle(groups)
        training, validation = [], []
        for group in groups:
            (training if len(training) + len(group) <= boundary else validation).extend(group)
        return sorted(training), sorted(validation)
    random.Random(SEED + ord(fold)).shuffle(ids)
    return sorted(ids[:boundary]), sorted(ids[boundary:])


def prepare() -> None:
    validate_layer_sources()
    subprocess.run(
        [sys.executable, str(ROOT / f"datasets/{DATASET}_official.py")],
        check=True,
        cwd=ROOT,
    )
    values = rows()
    split = []
    for fold in ("A", "B"):
        train_ids, validation_ids = split_ids(fold)
        train_set = set(train_ids)
        for row in fold_rows(fold):
            case = int(row["index"])
            split.append({
                "case_id": case,
                "fold": fold,
                "detector_partition": "train" if case in train_set else "validation",
            })
        write_json(DATA_DIR / f"fold_{fold.lower()}_split.json", {
            "fold": fold,
            "train_case_ids": train_ids,
            "validation_case_ids": validation_ids,
            "train_count": len(train_ids),
            "validation_count": len(validation_ids),
            "seed": SEED + ord(fold),
        })
    write_jsonl(DATA_DIR / "fold_split.jsonl", sorted(split, key=lambda item: item["case_id"]))
    config = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset": "RealWorldQA" if DATASET == "realworldqa" else "MMStar",
        "dataset_id": "xai-org/RealworldQA" if DATASET == "realworldqa" else "Lin-Chen/MMStar",
        "dataset_revision": ("17e7f75e092e47169732462ea3cdfebe911105dd" if DATASET == "realworldqa"
                             else "bc98d668301da7b14f648724866e57302778ab27"),
        "protocol": PROTOCOL,
        "prompt_source": str(MANIFEST),
        "prompt_field": "official question field verbatim",
        "fold_rule": ("seed 73 image-group shuffle; greedily fill A=383 then B=382" if DATASET == "realworldqa"
                      else "A=even case ID, B=odd case ID"),
        "directions": ["A_to_B", "B_to_A"],
        "layer": LAYER,
        "layer_source": "formal answer-state dynamics Composite validation pre-thinking ridge probe",
        "hook_site": "resid_pre",
        "resid_pre_decoder_index": LAYER,
        "equivalent_previous_block_output_index": LAYER - 1,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "detector_features": "current state, current minus previous checkpoint state, checkpoint/6144",
        "detector": "single linear layer",
        "threshold_source": "opposite held-out fold only: A selects B threshold; B selects A threshold",
        "maximum_validation_accuracy_drop": MAX_ACCURACY_DROP,
        "online_patience": 1,
        "thinking": "enabled",
        "decoding": "model-native paper generation config",
        "native_trace_backend": "sglang" if MODEL_KEY == "gemma12b" else "vllm",
        "hidden_state_replay_backend": "transformers",
        "adaptive_online_backend": "transformers",
        "generation_parameters": generation_parameters(),
        "forced_stop_transition": forced_stop_text(),
        "parser": (
            "realworldqa_answer_scoring_v1; final-only, gold-independent MC and atomic short-answer extraction"
            if DATASET == "realworldqa"
            else "benchmark_answer_scoring_v1; gold-independent final-answer assertion, no later option mention"
        ),
        "answer_scoring_version": (
            ANSWER_SCORING_VERSION
        ),
        "generation_token_limit": MAX_NEW_TOKENS,
        "model_context_length": MODEL_CONTEXT_LENGTH,
        "answer_generation_token_limit": ANSWER_MAX_TOKENS,
        "prompt_reasoning_budget_instruction": None,
        "initial_batch_size": INITIAL_BATCH_SIZE,
        "adaptive_batch_size": ADAPTIVE_BATCH_SIZE,
        "adaptive_batch_size_by_direction": {
            "A_to_B": ADAPTIVE_BATCH_SIZE,
            "B_to_A": (
                8
                if DATASET == "realworldqa" and MODEL_KEY == "qwen9b"
                else ADAPTIVE_BATCH_SIZE
            ),
        },
        "seed": SEED,
        "synthetic_smoke_test": False,
    }
    if DATASET == "realworldqa":
        from transformers import AutoProcessor, AutoTokenizer
        processor = AutoProcessor.from_pretrained(MODEL, revision=MODEL_REVISION)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
        processor_hash = chat_template_hash(processor)
        if processor_hash != chat_template_hash(tokenizer):
            raise SystemExit("RealWorldQA processor/tokenizer chat templates differ.")
        config["chat_template_sha256"] = processor_hash
        config["parser"] = "realworldqa_answer_scoring_v1; final-only, gold-independent MC and atomic short-answer extraction"
        config["evaluation_protocol"] = "internal early stopping; not VLMEvalKit leaderboard-comparable"
        config["aggregate_weighting"] = "held-out sample count (382/383)"
        config["split_metadata_sha256"] = hashlib.sha256((split_root / "split.json").read_bytes()).hexdigest()
        config["fold_file_sha256"] = SPLIT_METADATA["sha256"]
        config["duplicate_image_groups"] = IMAGE_GROUPS
        config["detector_split_rule"] = "image-group shuffle with seed 73+ord(fold), greedily fill floor(0.8*n) train"
    config_path = RESULT_DIR / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config and REQUIRE_ANSWER_SCORING_REFRESH:
            for key in ("parser", "answer_scoring_version"):
                existing[key] = config[key]
        if existing != config:
            raise SystemExit(f"Existing run config does not match the formal protocol: {config_path}")
    write_json(config_path, config)
    log(f"prepared {len(values)} {DATASET} rows with frozen folds")


def chat_template_hash(processor_or_tokenizer) -> str:
    template = processor_or_tokenizer.chat_template
    if isinstance(template, dict):
        if "default" in template:
            template = template["default"]
        elif len(template) == 1:
            template = next(iter(template.values()))
        else:
            raise SystemExit("Multiple chat templates without an unambiguous default.")
    if not template:
        raise SystemExit("Missing chat template for native/replay equality check.")
    return hashlib.sha256(json.dumps(template, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def verify_chat_template(processor_or_tokenizer) -> str:
    actual = chat_template_hash(processor_or_tokenizer)
    config = json.loads((RESULT_DIR / "run_config.json").read_text(encoding="utf-8"))
    if actual != config["chat_template_sha256"]:
        raise SystemExit("Loaded backend chat template differs from frozen RealWorldQA template.")
    return actual


def validate_layer_sources() -> None:
    for key, config in FORMAL_MODELS.items():
        if DATASET == "realworldqa" and key != MODEL_KEY:
            continue
        result_dir = ROOT / "paper/results" / config["result_slug"] / "decodability_usability_gap"
        summary = json.loads((result_dir / "run_summary.json").read_text(encoding="utf-8"))
        selection = json.loads((result_dir / "layer_selection.json").read_text(encoding="utf-8"))
        best_accuracy = max(float(row["validation_accuracy"]) for row in selection)
        selected = min(
            int(row["layer"])
            for row in selection
            if float(row["validation_accuracy"]) == best_accuracy
        )
        if int(summary["selected_layer"]) != selected or int(config["layer"]) != selected:
            raise SystemExit(
                f"Answer-state dynamics layer mismatch for {config['model']}: "
                f"config={config['layer']} summary={summary['selected_layer']} recomputed={selected}."
            )


def generation_parameters() -> dict[str, Any]:
    if MODEL_SPEC.family == "gemma":
        return {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_new_tokens": MAX_NEW_TOKENS,
        }
    return {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS}


def state_shard(fold: str, checkpoint: int) -> Path:
    return DATA_DIR / f"fold_{fold.lower()}_checkpoint_states" / f"checkpoint_{checkpoint:05d}.pt"


def forced_path(fold: str) -> Path:
    return DATA_DIR / f"fold_{fold.lower()}_forced_stops.jsonl"


def forced_rows(fold: str) -> list[dict[str, Any]]:
    path = forced_path(fold)
    values = read_jsonl(path) if path.exists() else []
    if any(
        row.get("protocol") != PROTOCOL
        or row.get("model") != MODEL
        or row.get("model_revision") != MODEL_REVISION
        for row in values
    ):
        raise SystemExit(f"Refusing incompatible forced-stop rows in {path}.")
    if DATASET != "realworldqa":
        by_case = {int(row["index"]): row for row in fold_rows(fold)}
        scored = []
        for source in values:
            row = dict(source)
            case = int(row["case_id"])
            prediction, rule = choice_answer(
                str(row["canonical_generation"]), question_text(by_case[case])
            )
            row.update({
                "prediction": prediction,
                "is_correct": prediction == str(by_case[case]["answer"]).strip().upper(),
                "extraction_rule": rule,
                "answer_scoring_version": ANSWER_SCORING_VERSION,
            })
            scored.append(row)
        return scored
    by_case = {int(row["index"]): row for row in fold_rows(fold)}
    scored = []
    for source in values:
        row = dict(source)
        case = int(row["case_id"])
        prediction = extract_row_choice(
            str(row["canonical_generation"]),
            by_case[case],
            require_closed_thinking=True,
        )
        row.update({
            "prediction": prediction,
            "is_correct": prediction == str(by_case[case]["answer"]).strip().upper(),
            "answer_scoring_version": ANSWER_SCORING_VERSION,
        })
        scored.append(row)
    return scored


def thought_open_text() -> str:
    return "<|channel>thought\n" if MODEL_SPEC.family == "gemma" else "<think>"


def thought_close_text() -> str:
    return "<channel|>" if MODEL_SPEC.family == "gemma" else "</think>"


def forced_stop_text() -> str:
    if MODEL_SPEC.family == "gemma":
        return f"<channel|><|channel>final\n{ANSWER_CUE}"
    return f"</think>\n{ANSWER_CUE}"


def token_ids_for(tokenizer, text: str) -> list[int]:
    values = [int(value) for value in tokenizer(text, add_special_tokens=False).input_ids]
    if not values:
        raise RuntimeError(f"Tokenizer produced no tokens for {text!r}.")
    return values


def resid_pre_module(model, layer: int = LAYER):
    layers = decoder_layers(model)
    if not 1 <= layer < len(layers):
        raise ValueError(f"Answer-state dynamics layer {layer} is invalid for {len(layers)} decoder blocks.")
    # hidden_states[layer] == output(layers[layer - 1]) == input(layers[layer]).
    return layers[layer]


def trim_generation(token_ids: list[int], eos_ids: set[int]) -> list[int]:
    for offset, token_id in enumerate(token_ids):
        if token_id in eos_ids:
            return token_ids[:offset]
    return token_ids


def eos_token_ids(bundle) -> set[int]:
    values = bundle.model.generation_config.eos_token_id
    if isinstance(values, int):
        return {values}
    return {int(value) for value in (values or [])}


def extract_choice(text: str, *, require_closed_thinking: bool = False) -> str:
    # Alternative letter extractor, kept for reference only: the pipeline
    # scores through extract_row_choice, which applies the shared answer parser.
    compact = text.upper()
    if require_closed_thinking and "</THINK>" not in compact:
        return ""
    if "</THINK>" in compact:
        compact = compact.rsplit("</THINK>", 1)[-1]
    compact = compact.replace("<|IM_END|>", "").strip()
    patterns = (
        r"<ANSWER>\s*(?:\*{1,2})?\s*\(?([A-D])\)?",
        r"(?:CORRECT\s+)?(?:ANSWER|OPTION|CHOICE)\s*(?:IS\s*)?[:\-]?\s*(?:\*{1,2})?\s*\(?([A-D])\)?(?:\*{1,2})?(?:\b|[.)])",
        r"(?m)^\s*(?:\*{1,2})?\s*\(?([A-D])\)?(?:\*{1,2})?\s*(?=$|[.)\s:])",
    )
    matches = [match for pattern in patterns for match in re.finditer(pattern, compact)]
    return max(matches, key=lambda match: match.start()).group(1) if matches else ""


def extract_row_choice(text: str, row: dict[str, Any], *, require_closed_thinking: bool = False) -> str:
    if DATASET == "realworldqa":
        is_mc = not (
            row.get("question_type") == "short_answer"
            or "Please answer directly with a single word or number." in question_text(row)
        )
        if "</think>" not in text.lower():
            return extract_direct_answer(text, is_mc=is_mc)
        if is_mc:
            return extract_realworldqa_choice(text, question_text(row))
        return extract_realworldqa_final(
            text,
            is_mc=False,
            require_closed=require_closed_thinking,
        )
    if require_closed_thinking and "</think>" not in text.lower():
        return ""
    choice, _rule = choice_answer(text, question_text(row))
    return choice


def cache_fold(fold: str) -> None:
    """Cache all checkpoint states with one teacher-forced replay per trajectory."""
    fold_values = fold_rows(fold)
    by_case = {int(row["index"]): row for row in fold_values}
    traces = trace_rows(fold)
    shard_dir = DATA_DIR / f"fold_{fold.lower()}_checkpoint_states"
    shard_dir.mkdir(parents=True, exist_ok=True)
    completed_checkpoints = {
        checkpoint
        for path in shard_dir.glob("checkpoint_*.pt")
        if (checkpoint := int(path.stem.rsplit("_", 1)[-1]))
        and valid_state_shard(path, fold, checkpoint, traces)
    }
    all_checkpoints = set(range(
        CHECKPOINT_INTERVAL,
        max(trace_reasoning_tokens(trace) for trace in traces.values()) + 1,
        CHECKPOINT_INTERVAL,
    ))
    missing_checkpoints = sorted(all_checkpoints - completed_checkpoints)
    if not missing_checkpoints:
        log(f"fold {fold} state cache complete; checking forced answers")
        generate_missing_forced_answers(
            fold, sorted(all_checkpoints), by_case, traces
        )
        validate_forced_answers(fold, all_checkpoints, traces)
        return
    case_dir = DATA_DIR / f"fold_{fold.lower()}_case_states"
    case_dir.mkdir(parents=True, exist_ok=True)
    bundle = ModelAdapter.load(MODEL_KEY)
    if DATASET == "realworldqa":
        verify_chat_template(bundle.processor)
    freeze_model(bundle.model)
    bundle.model.eval()
    for offset, case in enumerate(sorted(traces), start=1):
        total = trace_reasoning_tokens(traces[case])
        needed = [checkpoint for checkpoint in missing_checkpoints if checkpoint <= total]
        if not needed:
            continue
        case_path = case_dir / f"case_{case:04d}.pt"
        if case_path.exists():
            payload = torch.load(case_path, map_location="cpu", weights_only=True)
            if (
                int(payload.get("case_id", -1)) == case
                and [int(value) for value in payload["checkpoints"]] == needed
                and int(payload["states"].shape[0]) == len(needed)
                and payload["states"].ndim == 2
                and int(payload["states"].shape[1]) > 0
                and payload.get("protocol") == PROTOCOL
                and payload.get("model") == MODEL
                and payload.get("model_revision") == MODEL_REVISION
                and int(payload.get("layer", -1)) == LAYER
                and payload.get("hook_site") == "resid_pre"
            ):
                continue
        states = case_checkpoint_states(bundle, by_case[case], traces[case], needed)
        temporary = case_path.with_suffix(".tmp")
        torch.save({
            "case_id": case,
            "checkpoints": needed,
            "states": states.half(),
            "protocol": PROTOCOL,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "hook_site": "resid_pre",
        }, temporary)
        temporary.replace(case_path)
        log(f"fold {fold} trajectory states {offset}/{len(traces)} case={case} checkpoints={len(needed)}")
    assembled = {checkpoint: ([], []) for checkpoint in missing_checkpoints}
    for case in sorted(traces):
        case_path = case_dir / f"case_{case:04d}.pt"
        if not case_path.exists():
            continue
        payload = torch.load(case_path, map_location="cpu", weights_only=True)
        for checkpoint, state in zip(payload["checkpoints"], payload["states"]):
            checkpoint = int(checkpoint)
            if checkpoint in assembled:
                assembled[checkpoint][0].append(case)
                assembled[checkpoint][1].append(state)
    for checkpoint in missing_checkpoints:
        shard = state_shard(fold, checkpoint)
        case_ids, states = assembled[checkpoint]
        temporary = shard.with_suffix(".tmp")
        torch.save({
            "case_ids": case_ids,
            "checkpoint": checkpoint,
            "states": torch.stack(states),
            "protocol": PROTOCOL,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "hook_site": "resid_pre",
        }, temporary)
        temporary.replace(shard)
        log(f"fold {fold} assembled checkpoint {checkpoint}: {len(case_ids)} states")
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    generate_missing_forced_answers(fold, sorted(all_checkpoints), by_case, traces)
    validate_forced_answers(fold, all_checkpoints, traces)


def case_checkpoint_states(bundle, row, trace, checkpoints):
    inputs = bundle.prepare_inputs(
        [question_text(row)],
        [image_path_from_messages(row["messages"])],
        enable_thinking=True,
    )
    prompt_width = int(inputs["input_ids"].shape[1])
    full_prefix = reasoning_prefix_ids(trace, checkpoints[-1])
    positions = [
        prompt_width + int(trace["reasoning_token_indices"][checkpoint - 1])
        for checkpoint in checkpoints
    ]
    inputs = append_token_suffixes(bundle, inputs, [full_prefix])
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, layer_inputs):
        captured["states"] = layer_inputs[0][0, positions].detach().float().cpu()

    target_layer = resid_pre_module(bundle.model)
    handle = target_layer.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            bundle.forward(inputs, use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    return captured["states"]


def reasoning_prefix_ids(trace: dict[str, Any], reasoning_tokens: int) -> list[int]:
    indices = [int(index) for index in trace["reasoning_token_indices"]]
    if reasoning_tokens < 1 or reasoning_tokens > len(indices):
        raise ValueError(f"Invalid reasoning prefix length {reasoning_tokens}/{len(indices)}.")
    return [int(value) for value in trace["token_ids"][: indices[reasoning_tokens - 1] + 1]]


def valid_state_shard(path: Path, fold: str, checkpoint: int, traces: dict[int, dict[str, Any]]) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - an interrupted checkpoint must be rebuilt
        return False
    expected = [
        case for case in sorted(traces)
        if trace_reasoning_tokens(traces[case]) >= checkpoint
    ]
    return (
        int(payload.get("checkpoint", -1)) == checkpoint
        and payload.get("protocol") == PROTOCOL
        and payload.get("model") == MODEL
        and payload.get("model_revision") == MODEL_REVISION
        and int(payload.get("layer", -1)) == LAYER
        and payload.get("hook_site") == "resid_pre"
        and [int(value) for value in payload.get("case_ids", [])] == expected
        and payload.get("states", torch.empty(0)).ndim == 2
        and int(payload.get("states", torch.empty((0, 0))).shape[0]) == len(expected)
        and int(payload.get("states", torch.empty((0, 0))).shape[1]) > 0
    )


def generate_missing_forced_answers(fold, checkpoints, by_case, traces):
    existing_rows = forced_rows(fold)
    existing = {
        (int(row["case_id"]), int(row["checkpoint"]))
        for row in existing_rows
    }
    work = [
        (case, checkpoint)
        for case in sorted(traces)
        for checkpoint in checkpoints
        if checkpoint <= trace_reasoning_tokens(traces[case]) and (case, checkpoint) not in existing
    ]
    if not work:
        return
    bundle = ModelAdapter.load(MODEL_KEY)
    if DATASET == "realworldqa":
        verify_chat_template(bundle.processor)
    stop_ids = token_ids_for(bundle.tokenizer, forced_stop_text())
    work.sort(key=lambda item: len(reasoning_prefix_ids(traces[item[0]], item[1])))
    batch_size = INITIAL_BATCH_SIZE
    offset = 0
    while offset < len(work):
        prefix_length = len(reasoning_prefix_ids(traces[work[offset][0]], work[offset][1]))
        end = offset
        while (
            end < len(work)
            and end - offset < batch_size
            and len(reasoning_prefix_ids(traces[work[end][0]], work[end][1])) == prefix_length
        ):
            end += 1
        batch = work[offset:end]
        batch_rows = [by_case[case] for case, _checkpoint in batch]
        inputs = bundle.prepare_inputs(
            [question_text(row) for row in batch_rows],
            [image_path_from_messages(row["messages"]) for row in batch_rows],
            enable_thinking=True,
        )
        suffixes = [reasoning_prefix_ids(traces[case], checkpoint) + stop_ids for case, checkpoint in batch]
        inputs = append_token_suffixes(bundle, inputs, suffixes)
        prompt_width = int(inputs["input_ids"].shape[1])
        seed_all(SEED)
        try:
            with torch.inference_mode():
                sequences = bundle.model.generate(
                    **inputs,
                    **bundle.generation_config(True, ANSWER_MAX_TOKENS),
                )
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
            log(f"fold {fold} forced answers: OOM, reducing batch size to {batch_size}")
            continue
        saved = []
        eos_ids = eos_token_ids(bundle)
        for (case, checkpoint), sequence in zip(batch, sequences):
            token_ids = trim_generation(
                [int(value) for value in sequence[prompt_width:].tolist()], eos_ids
            )
            answer = bundle.tokenizer.decode(token_ids, skip_special_tokens=False)
            canonical = f"</think>\n{ANSWER_CUE}{answer}"
            prediction = extract_row_choice(canonical, by_case[case], require_closed_thinking=True)
            gold = str(by_case[case]["answer"]).upper()
            saved.append({
                "case_id": case,
                "fold": fold,
                "checkpoint": checkpoint,
                "gold": gold,
                "prediction": prediction,
                "is_correct": prediction == gold,
                "answer_generation": answer,
                "canonical_generation": canonical,
                "answer_token_count": len(token_ids),
                "batch_size": len(batch),
                "forced_stop_transition": forced_stop_text(),
                "generation_config": bundle.generation_provenance(True, ANSWER_MAX_TOKENS, SEED),
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "protocol": PROTOCOL,
            })
        append_jsonl(forced_path(fold), saved)
        offset = end
        log(f"fold {fold} forced answers {offset}/{len(work)}")
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def append_token_suffixes(
    bundle: ModelAdapter,
    inputs: dict[str, torch.Tensor],
    suffixes: list[list[int]],
) -> dict[str, torch.Tensor]:
    lengths = {len(suffix) for suffix in suffixes}
    if len(lengths) != 1:
        raise ValueError("Batched continuation suffixes must have equal lengths.")
    suffix = torch.tensor(suffixes, dtype=torch.long, device=bundle.device)
    return bundle.extend_inputs(inputs, suffix, torch.ones_like(suffix))


def validate_forced_answers(fold, checkpoints, traces) -> None:
    present = {
        (int(row["case_id"]), int(row["checkpoint"]))
        for row in forced_rows(fold)
    }
    expected = {
        (case, checkpoint)
        for case in traces
        for checkpoint in checkpoints
        if checkpoint <= trace_reasoning_tokens(traces[case])
    }
    if present != expected:
        raise SystemExit(
            f"Fold {fold} forced-stop cache mismatch: {len(present)}/{len(expected)} rows."
        )


def checkpoint_examples(fold: str) -> list[dict[str, Any]]:
    forced = {
        (int(row["case_id"]), int(row["checkpoint"])): row
        for row in forced_rows(fold)
    }
    states: dict[tuple[int, int], torch.Tensor] = {}
    directory = DATA_DIR / f"fold_{fold.lower()}_checkpoint_states"
    for path in sorted(directory.glob("checkpoint_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint = int(payload["checkpoint"])
        for case, state in zip(payload["case_ids"], payload["states"]):
            states[(int(case), checkpoint)] = state.float()
    by_case: dict[int, list[int]] = defaultdict(list)
    for case, checkpoint in states:
        by_case[case].append(checkpoint)
    examples = []
    for case, checkpoints in by_case.items():
        ordered = sorted(checkpoints)
        previous = None
        for checkpoint in ordered:
            current = states[(case, checkpoint)]
            current_outcome = forced[(case, checkpoint)]
            label = float(bool(current_outcome["is_correct"]))
            label_reason = "current_correct" if label else "current_wrong"
            previous_state = torch.zeros_like(current) if previous is None else previous
            examples.append({
                "case_id": case,
                "checkpoint": checkpoint,
                "feature": torch.cat((current, current - previous_state, torch.tensor([checkpoint / MAX_NEW_TOKENS]))),
                "label": label,
                "label_reason": label_reason,
                "forced_correct": bool(current_outcome["is_correct"]),
            })
            previous = current
    return examples


class LinearDetector(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.linear = nn.Linear(dimension, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values).squeeze(-1)


def direction_dir(train_fold: str, test_fold: str) -> Path:
    return RESULT_DIR / f"{train_fold}_to_{test_fold}"


def train_direction(train_fold: str, test_fold: str) -> None:
    output = direction_dir(train_fold, test_fold)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "detector.pt"
    selection_path = output / "validation_selection.json"
    if checkpoint_path.exists() and selection_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol") != PROTOCOL
            or payload.get("model") != MODEL
            or payload.get("model_revision") != MODEL_REVISION
            or int(payload.get("layer", -1)) != LAYER
            or payload.get("train_fold") != train_fold
            or payload.get("test_fold") != test_fold
            or selection.get("protocol") != PROTOCOL
            or selection.get("model") != MODEL
            or selection.get("model_revision") != MODEL_REVISION
            or int(selection.get("layer", -1)) != LAYER
            or selection.get("train_fold") != train_fold
            or selection.get("test_fold") != test_fold
        ):
            raise SystemExit(f"Refusing incompatible detector checkpoint in {output}.")
        scoring_matches = (
            not REQUIRE_ANSWER_SCORING_REFRESH
            or (
                payload.get("answer_scoring_version") == ANSWER_SCORING_VERSION
                and selection.get("answer_scoring_version") == ANSWER_SCORING_VERSION
            )
        )
        if scoring_matches:
            log(f"{train_fold}->{test_fold} detector already trained and selected")
            return
        log(f"{train_fold}->{test_fold} retraining detector for the current answer-scoring revision")
    examples = checkpoint_examples(train_fold)
    train_ids, validation_ids = split_ids(train_fold)
    train_set, validation_set = set(train_ids), set(validation_ids)
    labeled_train = [row for row in examples if row["case_id"] in train_set and row["label"] is not None]
    labeled_validation = [row for row in examples if row["case_id"] in validation_set and row["label"] is not None]
    if not labeled_train or not labeled_validation:
        raise SystemExit("Detector train/validation checkpoints are empty.")
    x_train = torch.stack([row["feature"] for row in labeled_train]).float()
    y_train = torch.tensor([row["label"] for row in labeled_train]).float()
    feature_mean = x_train.mean(0)
    feature_std = x_train.std(0).clamp_min(1e-5)
    x_train = (x_train - feature_mean) / feature_std
    detector = LinearDetector(x_train.shape[1])
    optimizer = torch.optim.AdamW(detector.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    positive = float(y_train.sum())
    negative = float(len(y_train) - positive)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / max(positive, 1.0)]))
    for epoch in range(TRAIN_EPOCHS):
        order = torch.randperm(len(x_train), generator=torch.Generator().manual_seed(SEED + epoch))
        for indices in order.split(256):
            logits = detector(x_train[indices])
            loss = criterion(logits, y_train[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    torch.save({
        "model_state": detector.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "input_dimension": x_train.shape[1],
        "train_fold": train_fold,
        "test_fold": test_fold,
        "train_case_ids": train_ids,
        "validation_case_ids": validation_ids,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "layer": LAYER,
        "hook_site": "resid_pre",
        "protocol": PROTOCOL,
        "answer_scoring_version": (
            ANSWER_SCORING_VERSION
        ),
    }, checkpoint_path)
    validation_probabilities = probabilities(detector, feature_mean, feature_std, examples, validation_set)
    selection = select_threshold(train_fold, validation_ids, validation_probabilities)
    selection.update({
        "protocol": PROTOCOL,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "layer": LAYER,
        "hook_site": "resid_pre",
        "train_fold": train_fold,
        "test_fold": test_fold,
        "detector_train_cases": len(train_ids),
        "detector_validation_cases": len(validation_ids),
        "labeled_train_checkpoints": len(labeled_train),
        "labeled_validation_checkpoints": len(labeled_validation),
        "positive_train_checkpoints": int(y_train.sum()),
        "labeled_checkpoints": len(examples),
        "train_validation_case_overlap": len(train_set & validation_set),
        "answer_scoring_version": (
            ANSWER_SCORING_VERSION
        ),
    })
    write_json(selection_path, selection)
    log(f"{train_fold}->{test_fold} selected threshold={selection['threshold']:.8f}")


def probabilities(detector, feature_mean, feature_std, examples, allowed_ids: set[int]) -> dict[tuple[int, int], float]:
    selected = [row for row in examples if row["case_id"] in allowed_ids]
    if not selected:
        return {}
    features = torch.stack([row["feature"] for row in selected]).float()
    with torch.inference_mode():
        values = torch.sigmoid(detector((features - feature_mean) / feature_std)).tolist()
    return {(row["case_id"], row["checkpoint"]): float(value) for row, value in zip(selected, values)}


def select_threshold(fold: str, validation_ids: list[int], probs: dict[tuple[int, int], float]) -> dict[str, Any]:
    traces = trace_rows(fold)
    by_case = {int(row["index"]): row for row in fold_rows(fold)}
    forced = {
        (int(row["case_id"]), int(row["checkpoint"])): row
        for row in forced_rows(fold)
    }
    candidates = sorted(set(probs.values()), reverse=True)
    candidates = [math.nextafter(max(candidates), math.inf)] + candidates + [0.0]
    full_accuracy = mean(
        extract_row_choice(
            str(traces[case]["generation"]), by_case[case], require_closed_thinking=True
        )
        == str(traces[case]["gold"]).upper()
        for case in validation_ids
    )
    feasible = []
    for threshold in candidates:
        outcomes = simulate_stopping(validation_ids, traces, forced, probs, threshold)
        accuracy = mean(float(row["is_correct"]) for row in outcomes)
        average_tokens = mean(float(row["reasoning_tokens"]) for row in outcomes)
        if accuracy >= full_accuracy - MAX_ACCURACY_DROP:
            feasible.append((average_tokens, -accuracy, -threshold, outcomes))
    if not feasible:
        raise RuntimeError("No validation threshold satisfies the frozen accuracy constraint.")
    average_tokens, negative_accuracy, negative_threshold, outcomes = min(feasible, key=lambda item: item[:3])
    return {
        "threshold": -negative_threshold,
        "validation_full_accuracy": full_accuracy,
        "validation_adaptive_accuracy": -negative_accuracy,
        "validation_accuracy_drop": full_accuracy + negative_accuracy,
        "validation_mean_reasoning_tokens": average_tokens,
        "validation_early_stop_fraction": mean(float(row["early_stopped"]) for row in outcomes),
        "candidate_threshold_count": len(candidates),
        "selection_objective": "minimum mean reasoning tokens subject to <=0.5pp full-thinking accuracy drop",
    }


def simulate_stopping(case_ids, traces, forced, probs, threshold) -> list[dict[str, Any]]:
    by_case = {int(row["index"]): row for row in rows()}
    output = []
    for case in case_ids:
        total = trace_reasoning_tokens(traces[case])
        stop = None
        for checkpoint in range(CHECKPOINT_INTERVAL, total + 1, CHECKPOINT_INTERVAL):
            if probs.get((case, checkpoint), -1.0) >= threshold:
                stop = checkpoint
                break
        if stop is None:
            prediction = extract_row_choice(
                str(traces[case]["generation"]),
                by_case[case],
                require_closed_thinking=True,
            )
            reasoning = total
        else:
            prediction = str(forced[(case, stop)]["prediction"])
            reasoning = stop
        gold = str(traces[case]["gold"]).upper()
        output.append({
            "case_id": case,
            "reasoning_tokens": reasoning,
            "prediction": prediction,
            "is_correct": prediction == gold,
            "early_stopped": stop is not None and stop < total,
        })
    return output


def load_detector(train_fold: str, test_fold: str, device) -> tuple[LinearDetector, torch.Tensor, torch.Tensor, float]:
    payload = torch.load(direction_dir(train_fold, test_fold) / "detector.pt", map_location="cpu", weights_only=True)
    if (
        payload.get("protocol") != PROTOCOL
        or payload.get("model") != MODEL
        or payload.get("model_revision") != MODEL_REVISION
        or int(payload.get("layer", -1)) != LAYER
        or payload.get("train_fold") != train_fold
        or payload.get("test_fold") != test_fold
        or (
            REQUIRE_ANSWER_SCORING_REFRESH
            and payload.get("answer_scoring_version") != ANSWER_SCORING_VERSION
        )
    ):
        raise SystemExit(f"Detector provenance mismatch for {train_fold}->{test_fold}.")
    detector = LinearDetector(int(payload["input_dimension"]))
    detector.load_state_dict(payload["model_state"])
    detector.to(device).eval()
    selection = json.loads((direction_dir(train_fold, test_fold) / "validation_selection.json").read_text(encoding="utf-8"))
    if (
        selection.get("protocol") != PROTOCOL
        or selection.get("model") != MODEL
        or selection.get("model_revision") != MODEL_REVISION
        or int(selection.get("layer", -1)) != LAYER
        or selection.get("train_fold") != train_fold
        or selection.get("test_fold") != test_fold
        or (
            REQUIRE_ANSWER_SCORING_REFRESH
            and selection.get("answer_scoring_version") != ANSWER_SCORING_VERSION
        )
    ):
        raise SystemExit(f"Threshold provenance mismatch for {train_fold}->{test_fold}.")
    return (
        detector,
        payload["feature_mean"].to(device),
        payload["feature_std"].to(device),
        float(selection["threshold"]),
    )


class OnlineStopController:
    def __init__(
        self,
        batch: int,
        open_ids: list[int],
        close_ids: list[int],
        stop_ids: list[int],
        eos_id: int,
        detector,
        mean,
        std,
        threshold,
    ):
        self.open_ids = open_ids
        self.close_ids = close_ids
        self.stop_ids = stop_ids
        self.eos_id = eos_id
        self.detector = detector
        self.mean = mean
        self.std = std
        self.threshold = threshold
        self.previous: list[torch.Tensor | None] = [None] * batch
        self.inject = [[] for _ in range(batch)]
        self.captured: torch.Tensor | None = None
        self.prompt_width: int | None = None
        self.processed_width = 0
        self.generated_counts = [0] * batch
        self.reasoning_counts = [0] * batch
        self.open_seen = [False] * batch
        self.closed = [False] * batch
        self.answer_start_at: list[int | None] = [None] * batch
        self.answer_start_pending = [False] * batch
        self.token_tails = [[] for _ in range(batch)]
        self.pattern_width = max(len(open_ids), len(close_ids))
        self.decisions = [[] for _ in range(batch)]
        self.forced_at: list[int | None] = [None] * batch

    def hook(self, _module, layer_inputs):
        self.captured = layer_inputs[0][:, -1].detach().float()

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.prompt_width is None:
            self.prompt_width = int(input_ids.shape[1])
            return scores
        generated_width = int(input_ids.shape[1]) - self.prompt_width
        if generated_width > self.processed_width:
            new_tokens = input_ids[:, self.prompt_width + self.processed_width :].detach().cpu().tolist()
            for index, values in enumerate(new_tokens):
                for value in values:
                    token = int(value)
                    self.generated_counts[index] += 1
                    tail = self.token_tails[index]
                    tail.append(token)
                    if len(tail) > self.pattern_width:
                        del tail[:-self.pattern_width]
                    if not self.open_seen[index] and tail[-len(self.open_ids) :] == self.open_ids:
                        self.open_seen[index] = True
                        self.reasoning_counts[index] = 0
                    elif self.open_seen[index]:
                        self.reasoning_counts[index] += 1
                    else:
                        self.reasoning_counts[index] = self.generated_counts[index]
                    if tail[-len(self.close_ids) :] == self.close_ids:
                        self.closed[index] = True
            self.processed_width = generated_width
        output = scores.clone()
        for index in range(input_ids.shape[0]):
            if self.answer_start_pending[index]:
                self.answer_start_at[index] = self.generated_counts[index]
                self.answer_start_pending[index] = False
            if self.inject[index]:
                force_token(output[index], self.inject[index].pop(0))
                if not self.inject[index]:
                    self.answer_start_pending[index] = True
                continue
            answer_start = self.answer_start_at[index]
            if answer_start is not None and self.generated_counts[index] - answer_start >= ANSWER_MAX_TOKENS:
                force_token(output[index], self.eos_id)
                continue
            if self.closed[index]:
                continue
            count = self.reasoning_counts[index]
            should_stop = False
            if count > 0 and count % CHECKPOINT_INTERVAL == 0:
                assert self.captured is not None
                current = self.captured[index]
                previous = torch.zeros_like(current) if self.previous[index] is None else self.previous[index]
                feature = torch.cat((current, current - previous, current.new_tensor([count / MAX_NEW_TOKENS])))
                probability = float(torch.sigmoid(self.detector((feature - self.mean) / self.std)).item())
                self.previous[index] = current
                should_stop = probability >= self.threshold
                self.decisions[index].append({"checkpoint": count, "probability": probability, "safe": probability >= self.threshold})
            if should_stop:
                self.forced_at[index] = count
                force_token(output[index], self.stop_ids[0])
                self.inject[index] = self.stop_ids[1:]
        return output


def force_token(scores: torch.Tensor, token_id: int) -> None:
    selected = scores[token_id].clone()
    scores.fill_(-float("inf"))
    scores[token_id] = selected


def find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    for start in range(len(values) - len(pattern) + 1):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None


def reasoning_count(values: list[int], open_ids: list[int]) -> int:
    opening = find_subsequence(values, open_ids)
    return len(values) if opening is None else len(values) - opening - len(open_ids)


def prediction_path(train_fold: str, test_fold: str, condition: str) -> Path:
    return direction_dir(train_fold, test_fold) / f"{condition}_predictions.jsonl"


def generate_condition(train_fold: str, test_fold: str, condition: str, *, shared_bundle=None) -> None:
    if condition not in {"full", "adaptive"}:
        raise ValueError(f"Unknown formal condition: {condition}")
    path = prediction_path(train_fold, test_fold, condition)
    done = latest_prediction_by_case(path)
    pending = [row for row in fold_rows(test_fold) if int(row["index"]) not in done]
    if not pending:
        return
    trace_lengths = trace_rows(test_fold)
    if condition == "full":
        materialize_full_predictions(train_fold, test_fold, pending, path, trace_lengths)
        return
    pending.sort(key=lambda row: trace_reasoning_tokens(trace_lengths[int(row["index"])]))
    bundle = shared_bundle if shared_bundle is not None else ModelAdapter.load(MODEL_KEY)
    if DATASET == "realworldqa":
        verify_chat_template(bundle.processor)
    freeze_model(bundle.model)
    bundle.model.eval()
    detector = feature_mean = feature_std = threshold = None
    if condition == "adaptive":
        detector, feature_mean, feature_std, threshold = load_detector(train_fold, test_fold, bundle.model.device)
    open_ids = token_ids_for(bundle.tokenizer, thought_open_text())
    close_ids = token_ids_for(bundle.tokenizer, thought_close_text())
    stop_ids = token_ids_for(bundle.tokenizer, forced_stop_text())
    eos_ids = eos_token_ids(bundle)
    batch_size = (
        8
        if DATASET == "realworldqa" and MODEL_KEY == "qwen9b" and train_fold == "B"
        else ADAPTIVE_BATCH_SIZE
    )
    offset = 0
    while offset < len(pending):
        batch_rows = pending[offset : offset + batch_size]
        inputs = bundle.prepare_inputs(
            [question_text(row) for row in batch_rows],
            [image_path_from_messages(row["messages"]) for row in batch_rows],
            enable_thinking=True,
        )
        prompt_width = int(inputs["input_ids"].shape[1])
        controller = OnlineStopController(
            len(batch_rows), open_ids, close_ids, stop_ids, min(eos_ids),
            detector, feature_mean, feature_std, threshold,
        )
        target_layer = resid_pre_module(bundle.model)
        handle = target_layer.register_forward_pre_hook(controller.hook) if condition == "adaptive" else None
        try:
            seed_all(SEED)
            with torch.inference_mode():
                sequences = bundle.model.generate(
                    **inputs,
                    **bundle.generation_config(True, MAX_NEW_TOKENS),
                    logits_processor=[controller],
                    use_cache=True,
                )
        except torch.cuda.OutOfMemoryError:
            if handle is not None:
                handle.remove()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
            log(f"{train_fold}->{test_fold} {condition}: OOM, reducing batch size to {batch_size}")
            continue
        finally:
            if handle is not None and handle.id in target_layer._forward_pre_hooks:
                handle.remove()
        saved = []
        for index, (row, sequence) in enumerate(zip(batch_rows, sequences)):
            token_ids = trim_generation([int(value) for value in sequence[prompt_width:].tolist()], eos_ids)
            trace = parsed_trace(bundle.tokenizer, token_ids)
            generation = trace["canonical_generation"]
            prediction = extract_row_choice(
                generation, row, require_closed_thinking=True
            )
            gold = str(row["answer"]).upper()
            forced_at = controller.forced_at[index] if trace["closed_thinking"] else None
            saved.append({
                "case_id": int(row["index"]),
                "index": str(row["index"]),
                "train_fold": train_fold,
                "test_fold": test_fold,
                "condition": condition,
                "prediction": generation,
                "extracted_choice": prediction,
                "gold": gold,
                "is_correct": prediction == gold,
                "generation_tokens": len(token_ids),
                "reasoning_tokens": len(trace["reasoning_token_indices"]),
                "closed_thinking": trace["closed_thinking"],
                "batch_size": len(batch_rows),
                "early_stopped": forced_at is not None,
                "forced_stop_checkpoint": forced_at,
                "detector_decisions": controller.decisions[index],
                "forced_stop_transition": forced_stop_text(),
                "generation_config": bundle.generation_provenance(True, MAX_NEW_TOKENS, SEED),
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "protocol": PROTOCOL,
                "answer_scoring_version": (
                    ANSWER_SCORING_VERSION
                ),
            })
        if SAVE_GENERATED_TOKEN_IDS:
            for record, sequence in zip(saved, sequences):
                record["generated_token_ids"] = trim_generation(
                    [int(value) for value in sequence[prompt_width:].tolist()], eos_ids
                )
        append_jsonl(path, saved)
        offset += len(batch_rows)
        log(f"{train_fold}->{test_fold} {condition}: {offset}/{len(pending)} batch={batch_size}")
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def materialize_full_predictions(train_fold, test_fold, pending, path, traces) -> None:
    by_case = {int(row["index"]): row for row in pending}
    saved = []
    for case in sorted(by_case):
        row = by_case[case]
        trace = traces[case]
        generation = str(trace["generation"])
        prediction = extract_row_choice(generation, row, require_closed_thinking=True)
        gold = str(row["answer"]).upper()
        saved.append({
            "case_id": case,
            "index": str(case),
            "train_fold": train_fold,
            "test_fold": test_fold,
            "condition": "full",
            "prediction": generation,
            "extracted_choice": prediction,
            "gold": gold,
            "is_correct": prediction == gold,
            "generation_tokens": int(trace["generation_tokens"]),
            "reasoning_tokens": trace_reasoning_tokens(trace),
            "closed_thinking": bool(trace["closed_thinking"]),
            "batch_size": int(trace["batch_size"]),
            "early_stopped": False,
            "forced_stop_checkpoint": None,
            "detector_decisions": [],
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "generation_config": trace["generation_config"],
            "protocol": PROTOCOL,
            "answer_scoring_version": (
                ANSWER_SCORING_VERSION
            ),
        })
    append_jsonl(path, saved)
    log(f"{train_fold}->{test_fold} full: materialized {len(saved)}/{FOLD_SIZES[test_fold]} formal traces")


def evaluate_direction_adaptive_only(train_fold: str, test_fold: str) -> None:
    generate_condition(train_fold, test_fold, "full")
    generate_condition(train_fold, test_fold, "adaptive")
    grouped = {
        condition: list(latest_prediction_by_case(
            prediction_path(train_fold, test_fold, condition)
        ).values())
        for condition in ("full", "adaptive")
    }
    if any(len(values) != FOLD_SIZES[test_fold] for values in grouped.values()):
        raise SystemExit("Full/adaptive held-out predictions are incomplete.")
    metrics = {
        condition: condition_metrics(values)
        for condition, values in grouped.items()
    }
    full = metrics["full"]
    adaptive = metrics["adaptive"]
    write_json(direction_dir(train_fold, test_fold) / "summary.json", {
        "train_fold": train_fold,
        "test_fold": test_fold,
        "test_count": FOLD_SIZES[test_fold],
        "conditions": metrics,
        "adaptive_accuracy_drop": full["accuracy"] - adaptive["accuracy"],
        "adaptive_reasoning_token_reduction": (
            1.0 - adaptive["mean_reasoning_tokens"] / full["mean_reasoning_tokens"]
        ),
    })

def condition_metrics(values: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [float(row["reasoning_tokens"]) for row in values]
    return {
        "n": len(values),
        "correct": sum(bool(row["is_correct"]) for row in values),
        "accuracy": mean(float(row["is_correct"]) for row in values),
        "mean_reasoning_tokens": mean(tokens),
    }


def summarize_external() -> None:
    directions = [
        json.loads((direction_dir(train, test) / "summary.json").read_text(encoding="utf-8"))
        for train, test in (("A", "B"), ("B", "A"))
    ]
    aggregate = {
        condition: {
            key: (sum(float(direction["conditions"][condition][key]) * direction["test_count"]
                      for direction in directions) / sum(direction["test_count"] for direction in directions)
                  if DATASET == "realworldqa" else
                  mean(float(direction["conditions"][condition][key]) for direction in directions))
            for key in (
                "accuracy",
                "mean_reasoning_tokens",
            )
        }
        for condition in ("full", "adaptive")
    }
    write_json(RESULT_DIR / "summary.json", {
        "protocol": PROTOCOL,
        "directions": directions,
        "two_fold_average": aggregate,
        "adaptive_accuracy_drop": aggregate["full"]["accuracy"] - aggregate["adaptive"]["accuracy"],
        "adaptive_reasoning_token_reduction": (
            1.0 - aggregate["adaptive"]["mean_reasoning_tokens"] / aggregate["full"]["mean_reasoning_tokens"]
        ),
    })


if __name__ == "__main__":
    main()
