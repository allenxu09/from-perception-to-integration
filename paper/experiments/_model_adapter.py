"""Model-specific tensor plumbing for the frozen paper experiments."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_core.models import build_vision_message, load_model


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    model_id: str
    revision: str
    result_slug: str
    batch_size_behavioral: int
    batch_size_readout: int
    batch_size_patching: int
    trace_batch_size: int
    load_in_8bit: bool = False
    fp8_dynamic: bool = False

    @property
    def quantized(self) -> bool:
        return self.load_in_8bit or self.fp8_dynamic


MODEL_SPECS = {
    "qwen": ModelSpec(
        "qwen",
        "qwen",
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "qwen3_5_4b",
        16,
        16,
        4,
        16,
    ),
    "gemma": ModelSpec(
        "gemma",
        "gemma",
        "google/gemma-4-E4B-it",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "gemma_4_e4b_it",
        1,
        1,
        1,
        8,
    ),
    "qwen9b": ModelSpec(
        "qwen9b",
        "qwen",
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "qwen3_5_9b_fp8_dynamic",
        16,
        16,
        16,
        16,
        False,
        True,
    ),
    "gemma12b": ModelSpec(
        "gemma12b",
        "gemma",
        "google/gemma-4-12B-it",
        "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "gemma_4_12b_it_fp8_dynamic",
        16,
        16,
        16,
        4,
        False,
        True,
    ),
}


def prequantized_path(spec: ModelSpec) -> Path:
    return Path(os.environ.get("VLM_CORE_QUANTIZED_ROOT", "quantized_models")) / f"{spec.result_slug}_bnb_8bit"


def fp8_checkpoint_path(spec: ModelSpec) -> Path:
    return Path(os.environ.get("VLM_CORE_QUANTIZED_ROOT", "quantized_models")) / spec.result_slug


def has_prequantized_checkpoint(spec: ModelSpec) -> bool:
    if spec.fp8_dynamic:
        return (fp8_checkpoint_path(spec) / "compression_manifest.json").is_file()
    return (prequantized_path(spec) / "export_manifest.json").is_file()


def generation_adapter(
    key: str,
    *,
    context_length: int = 8192,
    max_running_requests: int | None = None,
):
    spec = model_spec(key)
    if spec.fp8_dynamic and spec.family == "gemma":
        return SglangGenerationAdapter.load(
            key,
            context_length=context_length,
            max_running_requests=max_running_requests,
        )
    if spec.quantized and has_prequantized_checkpoint(spec):
        return VllmGenerationAdapter.load(key, max_model_len=context_length)
    return ModelAdapter.load(key)

THINKING_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:
1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.
Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()


def model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown paper model: {key}") from exc


def verify_8bit_load(model) -> dict[str, Any]:
    from bitsandbytes.nn import Linear4bit, Linear8bitLt

    config = getattr(model, "quantization_config", None) or getattr(model.config, "quantization_config", None)
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    config = config or {}
    report = {
        "model_is_loaded_in_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
        "model_is_loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        "config_load_in_8bit": config.get("load_in_8bit") is True,
        "config_load_in_4bit": config.get("load_in_4bit") is True,
        "linear8bitlt_modules": sum(isinstance(module, Linear8bitLt) for module in model.modules()),
        "linear4bit_modules": sum(isinstance(module, Linear4bit) for module in model.modules()),
    }
    if (
        not report["model_is_loaded_in_8bit"]
        or report["model_is_loaded_in_4bit"]
        or not report["config_load_in_8bit"]
        or report["config_load_in_4bit"]
        or not report["linear8bitlt_modules"]
        or report["linear4bit_modules"]
    ):
        raise ValueError(f"Strict bitsandbytes 8-bit verification failed: {report}")
    return report


def verify_fp8_load(model) -> dict[str, Any]:
    config = getattr(model.config, "quantization_config", {})
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    serialized = json.dumps(config).lower()
    report = {
        "quant_method": config.get("quant_method"),
        "format": config.get("format"),
        "fp8": "float" in serialized and "8" in serialized,
        "dynamic_activations": '"dynamic": true' in serialized,
    }
    if report["quant_method"] != "compressed-tensors" or not report["fp8"] or not report["dynamic_activations"]:
        raise ValueError(f"FP8_DYNAMIC verification failed: {report}")
    return report


class ModelAdapter:
    def __init__(self, spec: ModelSpec, model, processor=None, tokenizer=None):
        self.spec = spec
        self.model = model.eval()
        if spec.fp8_dynamic:
            self._quantization = verify_fp8_load(self.model)
        elif spec.load_in_8bit:
            self._quantization = verify_8bit_load(self.model)
        elif next(self.model.parameters()).dtype != torch.bfloat16:
            raise ValueError(f"{spec.model_id} did not load in bfloat16")
        else:
            self._quantization = None
        self.processor = processor
        self.tokenizer = tokenizer or getattr(processor, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError(f"{spec.model_id} did not provide a tokenizer")
        self.tokenizer.padding_side = "left"

    @classmethod
    def load(cls, key: str) -> "ModelAdapter":
        spec = model_spec(key)
        if spec.fp8_dynamic:
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            checkpoint = fp8_checkpoint_path(spec)
            if not (checkpoint / "compression_manifest.json").is_file():
                raise FileNotFoundError(f"Missing FP8 checkpoint: {checkpoint}")
            processor = AutoProcessor.from_pretrained(checkpoint)
            model = AutoModelForMultimodalLM.from_pretrained(
                checkpoint,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            return cls(spec, model, processor=processor)
        if spec.load_in_8bit:
            os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
            from unsloth import FastVisionModel

            exported = prequantized_path(spec)
            source = exported if (exported / "export_manifest.json").is_file() else spec.model_id
            model, processor = FastVisionModel.from_pretrained(
                model_name=str(source),
                revision=None if source == exported else spec.revision,
                max_seq_length=8192,
                load_in_8bit=True,
                load_in_4bit=False,
                use_exact_model_name=True,
                device_map="sequential",
            )
            FastVisionModel.for_inference(model)
            return cls(spec, model, processor=processor)

        bundle = load_model(
            spec.model_id,
            device="auto",
            attn_implementation="sdpa" if spec.family == "gemma" else None,
            revision=spec.revision,
            torch_dtype=torch.bfloat16,
        )
        return cls(spec, bundle.model, processor=bundle.processor, tokenizer=bundle.tokenizer)

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def result_root(self) -> Path:
        return Path(__file__).resolve().parents[1] / "results" / self.spec.result_slug

    def provenance(self) -> dict[str, Any]:
        report = {
            "model": self.spec.model_id,
            "model_revision": self.spec.revision,
            "adapter": self.spec.key,
            "precision": "fp8_dynamic_w8a8" if self.spec.fp8_dynamic else ("bitsandbytes_int8" if self.spec.load_in_8bit else "bfloat16"),
            "quantization": "compressed_tensors_fp8_dynamic" if self.spec.fp8_dynamic else ("bitsandbytes_8bit" if self.spec.load_in_8bit else "none"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "torchvision": importlib.metadata.version("torchvision"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        if self.spec.fp8_dynamic:
            report.update(
                llmcompressor=importlib.metadata.version("llmcompressor"),
                fp8_verification=self._quantization,
                checkpoint=str(fp8_checkpoint_path(self.spec)),
            )
        elif self.spec.load_in_8bit:
            report.update(
                unsloth=importlib.metadata.version("unsloth"),
                bitsandbytes=importlib.metadata.version("bitsandbytes"),
                load_in_8bit=True,
                load_in_4bit=False,
                eight_bit_verification=self._quantization,
            )
        return report

    def prepare_inputs(
        self,
        questions: list[str],
        images: list[str | Path | None],
        *,
        enable_thinking: bool,
    ) -> dict[str, torch.Tensor]:
        if len(questions) != len(images):
            raise ValueError("questions and images must have the same length")

        messages = []
        for question, image in zip(questions, images):
            if self.spec.family == "gemma":
                content = []
                if image is not None:
                    content.append({"type": "image", "url": str(image)})
                content.append({"type": "text", "text": question})
                messages.append([{"role": "user", "content": content}])
            else:
                messages.append(build_vision_message(question, image))
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)


    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        logits_to_keep: int = 1,
    ):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.spec.quantized and torch.cuda.is_available(),
        ):
            return self.model(
                **inputs,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
            )

    @torch.inference_mode()
    def generate(
        self,
        questions: list[str],
        images: list[str | Path | None],
        *,
        enable_thinking: bool,
        max_new_tokens: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        inputs = self.prepare_inputs(questions, images, enable_thinking=enable_thinking)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        config = self.generation_config(enable_thinking, max_new_tokens)
        output = self.model.generate(**inputs, **config)
        generated = output[:, inputs["input_ids"].shape[1] :]
        return [
            {
                "token_ids": [int(token) for token in row.tolist()],
                "text": self.tokenizer.decode(row, skip_special_tokens=False),
                "clean_text": self.tokenizer.decode(row, skip_special_tokens=True).strip(),
            }
            for row in generated
        ]

    def generation_config(self, enable_thinking: bool, max_new_tokens: int) -> dict[str, Any]:
        if not enable_thinking:
            return {"max_new_tokens": max_new_tokens, "do_sample": False}
        if self.spec.family == "gemma":
            return {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
            }
        return {"max_new_tokens": max_new_tokens, "do_sample": False}

    def generation_provenance(self, enable_thinking: bool, max_new_tokens: int, seed: int) -> dict[str, Any]:
        return {
            **self.generation_config(enable_thinking, max_new_tokens),
            "seed": seed,
            "backend": "transformers",
            "cache_implementation": "model_default_gpu",
        }

    def token_masks(self, inputs, questions: list[str], *, text_only: bool) -> dict[str, torch.Tensor]:
        input_ids = inputs["input_ids"]
        live = inputs["attention_mask"].bool()
        masks = {}
        if not text_only:
            image_id = self.image_token_id()
            image = (input_ids == image_id) & live
            if (image.sum(1) == 0).any():
                raise ValueError("Image token match failed")
            masks["image_tokens"] = image

        question = torch.zeros_like(live)
        for row, text in enumerate(questions):
            pattern = self.tokenizer(text, add_special_tokens=False).input_ids
            matches = [
                start
                for start in range(len(input_ids[row]) - len(pattern) + 1)
                if input_ids[row, start : start + len(pattern)].tolist() == pattern
            ]
            if len(matches) != 1:
                raise ValueError(f"Question token match failed: found {len(matches)} matches")
            question[row, matches[0] : matches[0] + len(pattern)] = True
        masks["question_tokens"] = question

        last = torch.zeros_like(live)
        positions = live.shape[1] - 1 - live.flip(1).int().argmax(dim=1)
        last[torch.arange(len(questions), device=live.device), positions] = True
        masks["last_prompt_token"] = last
        return masks

    def image_token_id(self) -> int:
        for owner in (self.processor, getattr(self.model, "config", None)):
            value = getattr(owner, "image_token_id", None)
            if value is not None:
                return int(value)
        return int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))

    def decoder_layers(self):
        paths = (
            ("model", "language_model", "layers"),
            ("language_model", "model", "layers"),
            ("model", "model", "layers"),
            ("model", "layers"),
            ("language_model", "layers"),
        )
        for path in paths:
            value = self.model
            for name in path:
                value = getattr(value, name, None)
                if value is None:
                    break
            if isinstance(value, torch.nn.ModuleList) and value:
                return value
        raise ValueError(f"Could not find decoder layers for {self.spec.model_id}")


    def final_norm(self):
        paths = (
            ("model", "language_model", "norm"),
            ("language_model", "model", "norm"),
            ("model", "model", "norm"),
            ("model", "norm"),
            ("language_model", "norm"),
        )
        for path in paths:
            value = self.model
            for name in path:
                value = getattr(value, name, None)
                if value is None:
                    break
            if isinstance(value, torch.nn.Module):
                return value
        raise ValueError(f"Could not find final norm for {self.spec.model_id}")

    def output_embeddings(self):
        return self.model.get_output_embeddings()

    def patched_forward(self, receiver_inputs, layer_index, donor_hidden, donor_mask, receiver_mask):
        layer = self.decoder_layers()[layer_index]

        def hook(_module, inputs):
            hidden = inputs[0].clone()
            values = []
            for row in range(len(hidden)):
                source = donor_hidden[row][donor_mask[row].to(donor_hidden.device)]
                if len(source) != int(receiver_mask[row].sum()):
                    raise ValueError("Donor and receiver token counts differ")
                values.append(source.to(hidden.device, dtype=hidden.dtype))
            hidden[receiver_mask] = torch.cat(values)
            return (hidden, *inputs[1:])

        handle = layer.register_forward_pre_hook(hook)
        try:
            return self.forward(receiver_inputs)
        finally:
            handle.remove()

    def extend_inputs(self, prompt, gen_ids: torch.Tensor, gen_mask: torch.Tensor):
        output = dict(prompt)
        prompt_len = output["input_ids"].shape[1]
        gen_ids = gen_ids.to(self.device)
        gen_mask = gen_mask.to(self.device)
        output["input_ids"] = torch.cat([output["input_ids"], gen_ids], dim=1)
        output["attention_mask"] = torch.cat([output["attention_mask"], gen_mask], dim=1)
        if "inputs_embeds" in output:
            extension = self.model.language_model.get_input_embeddings()(gen_ids)
            output["inputs_embeds"] = torch.cat([output["inputs_embeds"], extension], dim=1)
        for key, value in list(output.items()):
            if key in {"input_ids", "attention_mask", "inputs_embeds", "pixel_values"}:
                continue
            if value.ndim == 2 and value.shape[1] == prompt_len:
                extension = torch.zeros((value.shape[0], gen_ids.shape[1]), dtype=value.dtype, device=value.device)
                output[key] = torch.cat([value, extension], dim=1)
        return output

    def reasoning_token_data(self, token_ids: list[int], parsed_answer: str) -> dict[str, Any] | None:
        return reasoning_token_data(self.spec, self.tokenizer, token_ids, parsed_answer)


def verify_gemma4_vllm_patch() -> dict[str, Any]:
    version = importlib.metadata.version("vllm")
    expected = "0.27.2rc1.dev77+gac7509e2b.cu129"
    if version != expected:
        raise RuntimeError(f"Gemma 4 requires patched vLLM {expected}, found {version}")

    import vllm

    marker = Path(vllm.__file__).resolve().parent / "GEMMA4_PTH_PATCH.json"
    if not marker.is_file():
        raise RuntimeError(
            "Gemma 4 FP8 vLLM patch is missing; run "
            "`uv run python scripts/apply_vllm_gemma4_pth_patch.py`"
        )
    patch = json.loads(marker.read_text(encoding="utf-8"))
    if patch.get("pr_head") != "c74e90b9e2f457306f18ba593d6303e35fb560ec":
        raise RuntimeError(f"Unexpected Gemma 4 vLLM patch marker: {patch}")
    return {
        "kv_cache_dtype": "fp8_per_token_head",
        "vllm_gemma4_patch": patch,
    }


class SglangGenerationAdapter:
    SGLANG_COMMIT = "e161bd1265a0082478b7f1c09f224a52d315dc71"

    def __init__(self, spec: ModelSpec, engine, processor, manifest: dict[str, Any]):
        self.spec = spec
        self.engine = engine
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self._manifest = manifest

    @classmethod
    def load(
        cls,
        key: str,
        *,
        context_length: int = 8192,
        max_running_requests: int | None = None,
    ) -> SglangGenerationAdapter:
        from sglang import Engine
        from transformers import AutoProcessor

        spec = model_spec(key)
        checkpoint = fp8_checkpoint_path(spec)
        manifest = json.loads(
            (checkpoint / "compression_manifest.json").read_text(encoding="utf-8")
        )
        processor = AutoProcessor.from_pretrained(checkpoint)
        engine = Engine(
            model_path=str(checkpoint),
            model_impl="sglang",
            quantization="w8a8_fp8",
            dtype="auto",
            attention_backend="triton",
            sampling_backend="pytorch",
            context_length=context_length,
            max_running_requests=max_running_requests or spec.batch_size_behavioral,
            mem_fraction_static=0.85,
            enable_multimodal=True,
            disable_cuda_graph=True,
            random_seed=13,
        )
        return cls(spec, engine, processor, manifest)

    @property
    def result_root(self) -> Path:
        return Path(__file__).resolve().parents[1] / "results" / self.spec.result_slug

    def provenance(self) -> dict[str, Any]:
        return {
            **self._manifest,
            "backend": "sglang",
            "checkpoint": str(fp8_checkpoint_path(self.spec)),
            "sglang": importlib.metadata.version("sglang"),
            "sglang_commit": self.SGLANG_COMMIT,
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
            "quantization_runtime": "w8a8_fp8",
            "attention_backend": "triton",
            "sampling_backend": "pytorch",
            "disable_cuda_graph": True,
        }

    def generation_provenance(
        self, enable_thinking: bool, max_new_tokens: int, seed: int
    ) -> dict[str, Any]:
        return {
            **self.generation_config(enable_thinking, max_new_tokens),
            "sampling_seed": seed,
            "backend": "sglang",
            "chat_template_enable_thinking": enable_thinking,
        }

    def generation_config(
        self, enable_thinking: bool, max_new_tokens: int
    ) -> dict[str, Any]:
        if enable_thinking:
            return {
                "max_new_tokens": max_new_tokens,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
            }
        return {"max_new_tokens": max_new_tokens, "temperature": 0.0}

    def generate(
        self,
        questions,
        images,
        *,
        enable_thinking: bool,
        max_new_tokens: int,
        seed: int,
    ):
        prompts = []
        image_data = []
        for question, image in zip(questions, images):
            content = []
            if image is not None:
                content.append({"type": "image", "url": str(image)})
            content.append({"type": "text", "text": question})
            prompts.append(
                self.processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    tokenize=False,
                )
            )
            image_data.append(Image.open(image).convert("RGB") if image is not None else None)
        sampling = {
            **self.generation_config(enable_thinking, max_new_tokens),
            "sampling_seed": seed,
            "skip_special_tokens": False,
        }
        outputs = self.engine.generate(
            prompt=prompts,
            image_data=image_data,
            sampling_params=sampling,
        )
        rows = []
        for output in outputs:
            token_ids = [int(value) for value in output["output_ids"]]
            rows.append(
                {
                    "token_ids": token_ids,
                    "text": self.tokenizer.decode(
                        token_ids, skip_special_tokens=False
                    ),
                    "clean_text": self.tokenizer.decode(
                        token_ids, skip_special_tokens=True
                    ).strip(),
                }
            )
        return rows


class VllmGenerationAdapter:
    def __init__(
        self,
        spec: ModelSpec,
        llm,
        tokenizer,
        manifest: dict[str, Any],
        runtime: dict[str, Any],
    ):
        self.spec = spec
        self.llm = llm
        self.tokenizer = tokenizer
        self._manifest = manifest
        self._runtime = runtime

    @classmethod
    def load(cls, key: str, *, max_model_len: int = 8192) -> VllmGenerationAdapter:
        spec = model_spec(key)
        os.environ["FLA_GDN_FIX_BT"] = "1"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        from vllm import LLM

        checkpoint = fp8_checkpoint_path(spec) if spec.fp8_dynamic else prequantized_path(spec)
        manifest_name = "compression_manifest.json" if spec.fp8_dynamic else "export_manifest.json"
        manifest = json.loads((checkpoint / manifest_name).read_text(encoding="utf-8"))
        kwargs = dict(
            model=str(checkpoint),
            dtype="auto",
            max_model_len=max_model_len,
            max_num_seqs=spec.batch_size_behavioral,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=0.9,
        )
        runtime = {}
        if spec.fp8_dynamic and spec.family == "gemma":
            runtime = verify_gemma4_vllm_patch()
            kwargs["kv_cache_dtype"] = "fp8_per_token_head"
            kwargs["mm_processor_cache_gb"] = 0
            runtime["mm_processor_cache_gb"] = 0
        if not spec.fp8_dynamic:
            kwargs.update(quantization="bitsandbytes", load_format="bitsandbytes")
        llm = LLM(**kwargs)
        return cls(spec, llm, llm.get_tokenizer(), manifest, runtime)

    @classmethod
    def load_bfloat16(
        cls, key: str, *, max_model_len: int, max_num_seqs: int | None = None
    ) -> VllmGenerationAdapter:
        spec = model_spec(key)
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        from huggingface_hub import snapshot_download
        from vllm import LLM

        snapshot = snapshot_download(
            spec.model_id,
            revision=spec.revision,
            local_files_only=True,
        )
        kwargs = {
            "model": snapshot,
            "dtype": "bfloat16",
            "max_model_len": max_model_len,
            "max_num_seqs": spec.trace_batch_size if max_num_seqs is None else max_num_seqs,
            "limit_mm_per_prompt": {"image": 1},
            "gpu_memory_utilization": 0.9,
        }
        llm = LLM(**kwargs)
        manifest = {
            "model": spec.model_id,
            "model_revision": spec.revision,
            "precision": "bfloat16",
            "quantization": "none",
        }
        return cls(spec, llm, llm.get_tokenizer(), manifest, {})

    @property
    def result_root(self) -> Path:
        return Path(__file__).resolve().parents[1] / "results" / self.spec.result_slug

    def provenance(self) -> dict[str, Any]:
        return {
            **self._manifest,
            "backend": "vllm",
            "checkpoint": str(fp8_checkpoint_path(self.spec) if self.spec.fp8_dynamic else prequantized_path(self.spec)),
            "vllm": importlib.metadata.version("vllm"),
            **self._runtime,
        }

    def generation_provenance(self, enable_thinking: bool, max_new_tokens: int, seed: int) -> dict[str, Any]:
        return {
            **self.generation_config(enable_thinking, max_new_tokens),
            "seed": seed,
            "backend": "vllm",
            **self._runtime,
        }

    def generation_config(self, enable_thinking: bool, max_new_tokens: int) -> dict[str, Any]:
        if enable_thinking and self.spec.family == "gemma":
            return {"max_tokens": max_new_tokens, "temperature": 1.0, "top_p": 0.95, "top_k": 64}
        return {"max_tokens": max_new_tokens, "temperature": 0.0}

    def generate(self, questions, images, *, enable_thinking: bool, max_new_tokens: int, seed: int):
        from vllm import SamplingParams

        opened = [Image.open(path).convert("RGB") if path is not None else None for path in images]
        messages = []
        for question, image in zip(questions, opened):
            content = []
            if image is not None:
                content.append({"type": "image_pil", "image_pil": image})
            content.append({"type": "text", "text": question})
            messages.append([{"role": "user", "content": content}])
        try:
            outputs = self.llm.chat(
                messages,
                SamplingParams(seed=seed, **self.generation_config(enable_thinking, max_new_tokens)),
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )
        finally:
            for image in opened:
                if image is not None:
                    image.close()
        rows = []
        for output in outputs:
            token_ids = [int(value) for value in output.outputs[0].token_ids]
            rows.append(
                {
                    "token_ids": token_ids,
                    "text": self.tokenizer.decode(token_ids, skip_special_tokens=False),
                    "clean_text": self.tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                }
            )
        return rows

def find_subsequence(values: list[int], pattern: list[int], *, start: int = 0, last: bool = False) -> int | None:
    if not pattern:
        return None
    found = [index for index in range(start, len(values) - len(pattern) + 1) if values[index : index + len(pattern)] == pattern]
    if not found:
        return None
    return found[-1] if last else found[0]


def reasoning_token_data(spec: ModelSpec, tokenizer, token_ids: list[int], parsed_answer: str) -> dict[str, Any] | None:
    span = reasoning_span(spec, tokenizer, token_ids)
    if span is None:
        return None
    encode = lambda text: list(tokenizer(text, add_special_tokens=False).input_ids)
    final_start = span["final_start"]
    thought_indices = span["thought_token_indices"]
    final_indices = list(range(final_start, len(token_ids)))
    matches = []
    for text in (parsed_answer, f" {parsed_answer}"):
        pattern = encode(text)
        position = find_subsequence(token_ids, pattern, start=final_start, last=True)
        if position is not None:
            matches.append(list(range(position, position + len(pattern))))
    answer_indices = max(matches, key=lambda indices: indices[0]) if matches else []
    if not answer_indices or answer_indices[0] == 0:
        return None
    return {
        "thought_token_indices": thought_indices,
        "final_token_indices": final_indices,
        "answer_token_indices": answer_indices,
    }


def reasoning_span(
    spec: ModelSpec, tokenizer, token_ids: list[int], *, min_thought_tokens: int = 6
) -> dict[str, Any] | None:
    encode = lambda text: list(tokenizer(text, add_special_tokens=False).input_ids)
    if spec.family == "gemma":
        thought_open = encode("<|channel>thought\n")
        thought_close = encode("<channel|>")
        final_open = encode("<|channel>final\n")
    else:
        thought_open = encode("<think>")
        thought_close = encode("</think>")
        final_open = []
    close_start = find_subsequence(token_ids, thought_close)
    if close_start is None:
        return None
    open_start = find_subsequence(token_ids, thought_open)
    thought_start = open_start + len(thought_open) if open_start is not None and open_start < close_start else 0
    thought_indices = list(range(thought_start, close_start))
    final_start = close_start + len(thought_close)
    found_final = find_subsequence(token_ids, final_open, start=final_start) if final_open else None
    if found_final is not None:
        final_start = found_final + len(final_open)
    if len(thought_indices) < min_thought_tokens:
        return None
    return {
        "thought_token_indices": thought_indices,
        "final_start": final_start,
    }


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# Official InternVL3.5 image preprocessing from the model card.


