"""Small batched inference runner."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from .models import generate_answers
from .schema import DiagnosticSample, ModelBundle, Prediction


def batched(rows: Iterable[DiagnosticSample], batch_size: int) -> Iterator[list[DiagnosticSample]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    batch: list[DiagnosticSample] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_batched_generation(
    bundle: ModelBundle,
    samples: Iterable[DiagnosticSample],
    prompt_fn: Callable[[str], str],
    batch_size: int = 8,
    max_new_tokens: int = 40,
) -> Iterator[Prediction]:
    for batch in batched(samples, batch_size):
        answers = generate_answers(
            bundle,
            [prompt_fn(sample.question) for sample in batch],
            [sample.image_path or None for sample in batch],
            max_new_tokens=max_new_tokens,
        )
        for sample, answer in zip(batch, answers):
            yield Prediction(sample_id=sample.sample_id, answer=answer, raw_output=answer)
