"""Generate frozen native-thinking traces for answer-state dynamics."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from _model_adapter import (  # noqa: E402
    ModelAdapter,
    THINKING_SYSTEM_PROMPT,
    generation_adapter,
    has_prequantized_checkpoint,
    fp8_checkpoint_path,
    model_spec,
    prompt_fingerprint,
    reasoning_token_data,
)
from vlm_core.io import append_jsonl, read_jsonl, write_json  # noqa: E402


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
SEED = 13
MAX_NEW_TOKENS = 6144
GPU_MEMORY_UTILIZATION = 0.9
SUBMISSION_WINDOW = 128


def main(model_key: str = "qwen") -> None:
    spec = model_spec(model_key)
    result_dir = ROOT / "paper/results" / spec.result_slug / "decodability_usability_gap"
    trace_path = result_dir / "native_thinking_traces.jsonl"
    provenance_path = result_dir / "native_thinking_provenance.json"
    manifest = read_manifest()
    samples = read_samples()
    result_dir.mkdir(parents=True, exist_ok=True)

    existing = {str(row["sample_id"]): row for row in read_jsonl(trace_path)} if trace_path.exists() else {}
    unknown = set(existing) - {row["sample_id"] for row in samples}
    if unknown:
        raise SystemExit(f"Trace checkpoint contains unknown samples: {sorted(unknown)[:3]}")
    pending = [row for row in samples if row["sample_id"] not in existing]

    environment = vllm_environment(spec)
    generation_config = vllm_generation_config(spec)
    check_or_write_provenance(
        provenance_path,
        trace_provenance(manifest, samples, spec, environment, generation_config),
        trace_path,
    )
    generate_vllm(spec, samples, pending, trace_path, existing)

    provenance = trace_provenance(manifest, samples, spec, environment, generation_config)
    check_or_write_provenance(provenance_path, provenance, trace_path)
    write_summary(result_dir, samples, existing)


def vllm_environment(spec):
    if spec.fp8_dynamic:
        return json.loads((fp8_checkpoint_path(spec) / "compression_manifest.json").read_text(encoding="utf-8"))
    if spec.load_in_8bit:
        return json.loads(
            (ROOT / "paper/results" / spec.result_slug / "interface_check.json").read_text(encoding="utf-8")
        )
    return {
        "model": spec.model_id,
        "model_revision": spec.revision,
        "adapter": spec.key,
        "precision": "bfloat16",
        "quantization": "none",
        "transformers": importlib.metadata.version("transformers"),
        "vllm": importlib.metadata.version("vllm"),
    }


def vllm_generation_config(spec):
    backend = "vllm"
    if spec.fp8_dynamic:
        backend = (
            "transformers_compressed_tensors_fp8_dynamic"
            if spec.family == "gemma"
            else "vllm_compressed_tensors_fp8_dynamic"
        )
    elif has_prequantized_checkpoint(spec):
        backend = "vllm_prequantized_bnb_8bit"
    elif spec.load_in_8bit:
        backend = "transformers_unsloth"
    config = {
        "backend": backend,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "batch_size": spec.trace_batch_size,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    }
    if spec.fp8_dynamic:
        config.update(weight_precision="fp8", activation_precision="fp8_dynamic")
        if spec.family == "gemma":
            config["cache_implementation"] = "model_default_gpu"
    elif spec.load_in_8bit:
        config.update(load_in_8bit=True, load_in_4bit=False)
    if spec.family == "gemma":
        return {**config, "temperature": 1.0, "top_p": 0.95, "top_k": 64}
    return {**config, "temperature": 0.0}


def generate_vllm(spec, samples, pending, trace_path, existing):
    if not pending:
        return
    if spec.quantized:
        adapter = generation_adapter(spec.key)
        submission_window = spec.trace_batch_size if isinstance(adapter, ModelAdapter) else SUBMISSION_WINDOW
        for start in range(0, len(pending), submission_window):
            batch = pending[start : start + submission_window]
            outputs = adapter.generate(
                [prompt_for_spec(spec) for _ in batch],
                [row["image_path"] for row in batch],
                enable_thinking=True,
                max_new_tokens=MAX_NEW_TOKENS,
                seed=SEED + start,
            )
            saved = [
                trace_row(spec, adapter.tokenizer, sample, output["token_ids"], output["text"])
                for sample, output in zip(batch, outputs)
            ]
            append_jsonl(trace_path, saved)
            existing.update((row["sample_id"], row) for row in saved)
            print(f"[state dynamics traces/{spec.key}] {len(existing)}/{len(samples)}", flush=True)
        return

    from transformers import AutoProcessor
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    model_path = spec.model_id
    revision = spec.revision
    if spec.family == "gemma":
        model_path = snapshot_download(spec.model_id, revision=spec.revision, local_files_only=True)
        revision = None
    engine_kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        max_model_len=MAX_NEW_TOKENS + 2048,
        max_num_seqs=spec.trace_batch_size,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    if revision is not None:
        engine_kwargs["revision"] = revision
    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()
    if spec.family == "qwen":
        processor = AutoProcessor.from_pretrained(spec.model_id, revision=spec.revision)
        verify_chat_template(processor, tokenizer, pending[0])

    for start in range(0, len(pending), spec.trace_batch_size):
        batch = pending[start : start + spec.trace_batch_size]
        images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
        sampling_kwargs = {"temperature": 0.0, "max_tokens": MAX_NEW_TOKENS, "seed": SEED + start}
        if spec.family == "gemma":
            sampling_kwargs.update(temperature=1.0, top_p=0.95, top_k=64)
        sampling = SamplingParams(**sampling_kwargs)
        try:
            outputs = llm.chat(
                [vllm_message(spec.key, image) for image in images],
                sampling,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": True},
            )
        finally:
            for image in images:
                image.close()
        saved = []
        for sample, output in zip(batch, outputs):
            token_ids = [int(value) for value in output.outputs[0].token_ids]
            generation = tokenizer.decode(token_ids, skip_special_tokens=False)
            saved.append(trace_row(spec, tokenizer, sample, token_ids, generation))
        append_jsonl(trace_path, saved)
        existing.update((row["sample_id"], row) for row in saved)
        print(f"[state dynamics traces/{spec.key}] {len(existing)}/{len(samples)}", flush=True)


def trace_row(spec, tokenizer, sample, token_ids, generation):
    think_text, final_text, closed = split_reasoning(spec.family, generation)
    parsed = parse_choice(final_text) if closed else None
    token_data = reasoning_token_data(spec, tokenizer, token_ids, parsed) if parsed else None
    return {
        "sample_id": sample["sample_id"],
        "pair_id": sample["pair_id"],
        "split": sample["split"],
        "answer": sample["answer"],
        "larger_target_side": sample["larger_target_side"],
        "full_generation": generation,
        "generation_token_ids": token_ids,
        "think_text": think_text,
        "final_text": final_text,
        "reasoning_token_data": token_data,
        "parsed_answer": parsed or "",
        "parse_success": parsed is not None,
        "checkpoint_success": token_data is not None,
        "is_correct": parsed == sample["answer"],
        "generation_tokens": len(token_ids),
    }


def read_manifest() -> dict:
    return json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))


def read_samples() -> list[dict]:
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
                "split": split_by_id[row["sample_id"]],
                "image_path": str(image_path),
                "answer": str(row["answer"]),
                "larger_target_side": str(row["metadata"]["larger_target_side"]),
            }
        )
    return samples


def vllm_message(model_key: str, image) -> list[dict]:
    user = {
        "role": "user",
        "content": [{"type": "image_pil", "image_pil": image}, {"type": "text", "text": PROMPT}],
    }
    return [user]




def verify_chat_template(processor, tokenizer, sample: dict) -> None:
    placeholder = [{"role": "user", "content": [{"type": "image", "image": sample["image_path"]}, {"type": "text", "text": PROMPT}]}]
    kwargs = {"add_generation_prompt": True, "enable_thinking": True, "tokenize": False}
    if processor.apply_chat_template(placeholder, **kwargs) != tokenizer.apply_chat_template(placeholder, **kwargs):
        raise SystemExit("vLLM and Transformers chat templates differ")


def split_reasoning(model_key: str, text: str) -> tuple[str, str, bool]:
    if model_key == "gemma":
        match = re.search(r"<\|channel>thought\n(.*?)<channel\|>", text, flags=re.DOTALL)
        if not match:
            return text.strip(), "", False
        final = text[match.end() :]
        final = re.sub(r"^<\|channel>final\n", "", final).strip()
        return match.group(1).strip(), final, True
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), text[match.end() :].strip(), True
    end = re.search(r"</think>", text, flags=re.IGNORECASE)
    return (text[: end.start()].strip(), text[end.end() :].strip(), True) if end else (text.strip(), "", False)


def parse_choice(text: str) -> str | None:
    patterns = (
        r"\\boxed\{\s*([AB])\s*\}",
        r"(?:final answer|answer|option|choice)\s*(?:is|:|should be|would be)?\s*\**\s*([AB])\b",
        r"\b([AB])\s*\.\s*(?:left|right)\b",
        r"^\s*\*{1,2}\s*([AB])\s*\*{1,2}\s*\.?\s*$",
        r"^\s*([AB])\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()
    return None


def trace_provenance(manifest, samples, spec, environment, generation_config):
    prompt = prompt_for_spec(spec)
    return {
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["version"],
        "dataset_fingerprint": manifest["fingerprint"],
        "sample_ids": [row["sample_id"] for row in samples],
        **environment,
        "prompt": prompt,
        "prompt_fingerprint": prompt_fingerprint(prompt),
        "prompt_protocol": "thinking_switch_v3" if spec.quantized else PROMPT_PROTOCOL,
        "generation_config": generation_config,
        "enable_thinking": True,
    }


def prompt_for_spec(spec) -> str:
    return PROMPT_OFFICIAL_SWITCH if spec.quantized else PROMPT


def check_or_write_provenance(path: Path, expected: dict, trace_path: Path) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise SystemExit("Existing answer-state dynamics trace provenance does not match the canonical protocol")
    elif trace_path.exists():
        raise SystemExit("answer-state dynamics trace exists without provenance")
    else:
        write_json(path, expected)


def write_summary(result_dir: Path, samples: list[dict], rows: dict[str, dict]) -> None:
    ordered = [rows[row["sample_id"]] for row in samples if row["sample_id"] in rows]
    write_json(
        result_dir / "native_thinking_summary.json",
        {
            "expected_samples": len(samples),
            "completed_samples": len(ordered),
            "parse_success_samples": sum(bool(row["parse_success"]) for row in ordered),
            "checkpoint_success_samples": sum(bool(row.get("checkpoint_success")) for row in ordered),
            "correct_samples": sum(bool(row["is_correct"]) for row in ordered),
            "trace_path": str(result_dir / "native_thinking_traces.jsonl"),
        },
    )


if __name__ == "__main__":
    main()
