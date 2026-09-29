"""Compress the two formal VLMs to llm-compressor FP8 W8A8 checkpoints."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODELS = (
    (
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        Path(os.environ.get("VLM_CORE_QUANTIZED_ROOT", "quantized_models") + "/qwen3_5_9b_fp8_dynamic"),
    ),
    (
        "google/gemma-4-12B-it",
        "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        Path(os.environ.get("VLM_CORE_QUANTIZED_ROOT", "quantized_models") + "/gemma_4_12b_it_fp8_dynamic"),
    ),
)


def normalize_gemma4_config(output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["audio_config"].update(
        audio_samples_per_token=640,
        hidden_size=640,
        output_proj_dims=640,
    )
    config["vision_config"]["model_patch_size"] = 48
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    for model_id, revision, output_dir in MODELS:
        compress_model(model_id, revision, output_dir)


def compress_model(model_id: str, revision: str, output_dir: Path) -> None:
    manifest_path = output_dir / "compression_manifest.json"
    if manifest_path.is_file():
        if model_id == "google/gemma-4-12B-it":
            normalize_gemma4_config(output_dir)
        print(f"already compressed {model_id} -> {output_dir}", flush=True)
        return
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=[
            "re:.*lm_head",
            "re:.*visual.*",
            "re:.*vision.*",
            "re:.*audio.*",
            "re:.*mlp.gate$",
            "re:.*router$",
        ],
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    oneshot(model=model, recipe=recipe)
    model.save_pretrained(output_dir, save_compressed=True, safe_serialization=True)
    processor.save_pretrained(output_dir)
    if model_id == "google/gemma-4-12B-it":
        normalize_gemma4_config(output_dir)
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config", {})
    serialized = json.dumps(quantization).lower()
    if quantization.get("quant_method") != "compressed-tensors" or "dynamic" not in serialized:
        raise RuntimeError(f"Invalid FP8_DYNAMIC export: {quantization}")
    manifest_path.write_text(
        json.dumps(
            {
                "model": model_id,
                "model_revision": revision,
                "precision": "fp8_dynamic_w8a8",
                "quantization": "compressed_tensors_fp8_dynamic",
                "weight_quantization": "fp8_static_per_channel",
                "activation_quantization": "fp8_dynamic_per_token",
                "ignored": list(recipe.ignore),
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("llmcompressor", "compressed-tensors", "transformers", "torch")
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"compressed {model_id} -> {output_dir}", flush=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
