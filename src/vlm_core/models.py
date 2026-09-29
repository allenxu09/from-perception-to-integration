"""Model loading and generation helpers."""

from __future__ import annotations

from pathlib import Path
from time import sleep
from typing import Any

from .schema import ModelBundle

DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-4B"


def load_vllm_model(model_name: str = DEFAULT_QWEN_MODEL, **kwargs):
    """Load a vLLM model without importing vLLM in CPU-only workflows."""
    from vllm import LLM

    return LLM(model=model_name, **kwargs)


def load_model(
    model_name: str = DEFAULT_QWEN_MODEL,
    device: str = "auto",
    attn_implementation: str | None = None,
    revision: str | None = None,
    torch_dtype: Any = "auto",
) -> ModelBundle:
    name = model_name.lower()

    if "qwen" in name or "gemma" in name:
        return load_transformers_multimodal(
            model_name,
            device=device,
            attn_implementation=attn_implementation,
            revision=revision,
            torch_dtype=torch_dtype,
        )
    if "llava" in name:
        raise NotImplementedError("LLaVA loading needs transformers setup before first run.")

    raise ValueError(f"Unsupported model: {model_name!r} on device {device!r}")


def load_transformers_multimodal(
    model_name: str = DEFAULT_QWEN_MODEL,
    device: str = "auto",
    attn_implementation: str | None = None,
    revision: str | None = None,
    torch_dtype: Any = "auto",
) -> ModelBundle:
    transformers = _require_transformers()
    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if device == "auto":
        kwargs["device_map"] = "auto"
    if attn_implementation:
        kwargs["attn_implementation"] = attention_backend(model_name, attn_implementation)

    source_kwargs = {"revision": revision} if revision else {}
    processor = _from_pretrained(transformers.AutoProcessor, model_name, **source_kwargs)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    model = _from_pretrained(transformers.AutoModelForMultimodalLM, model_name, **source_kwargs, **kwargs)
    if device != "auto":
        model = model.to(device)

    return ModelBundle(
        name=model_name,
        model=model,
        processor=processor,
        tokenizer=getattr(processor, "tokenizer", None),
        device=device,
    )


def attention_backend(model_name: str, requested: str):
    if requested == "flash_attention_2" and "gemma" in model_name.lower():
        return {"text_config": requested, "vision_config": "sdpa", "audio_config": "sdpa"}
    return requested


def build_vision_message(question: str, image: str | Path | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = []
    if image is not None:
        image_text = str(image)
        key = "url" if image_text.startswith(("http://", "https://")) else "image"
        content.append({"type": "image", key: image_text})
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def generate_answer(
    bundle: ModelBundle,
    question: str,
    image: str | Path | None = None,
    max_new_tokens: int = 40,
) -> str:
    if bundle.processor is None:
        raise ValueError("ModelBundle.processor is required for generation")

    messages = build_vision_message(question, image=image)
    inputs = bundle.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(bundle.model.device)
    outputs = bundle.model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return bundle.processor.decode(generated, skip_special_tokens=True).strip()


def generate_answers(
    bundle: ModelBundle,
    questions: list[str],
    images: list[str | Path | None] | None = None,
    max_new_tokens: int = 40,
) -> list[str]:
    if bundle.processor is None:
        raise ValueError("ModelBundle.processor is required for generation")
    if images is None:
        images = [None] * len(questions)
    if len(questions) != len(images):
        raise ValueError("questions and images must have the same length")

    messages = [
        build_vision_message(question, image=image)
        for question, image in zip(questions, images)
    ]
    inputs = bundle.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(bundle.model.device)
    outputs = bundle.model.generate(**inputs, max_new_tokens=max_new_tokens)
    prompt_len = inputs["input_ids"].shape[-1]
    return [
        bundle.processor.decode(output[prompt_len:], skip_special_tokens=True).strip()
        for output in outputs
    ]


def _require_transformers():
    try:
        import transformers
    except ImportError as exc:
        raise SystemExit("Install model dependencies first: python -m pip install -e .") from exc
    return transformers


def _from_pretrained(loader, model_name: str, **kwargs):
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return loader.from_pretrained(model_name, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 2:
                break
            sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def freeze_model(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
