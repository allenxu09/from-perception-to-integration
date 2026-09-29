"""Freeze a random two-fold partition of the official RealWorldQA test rows."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from vlm_core.io import write_json, write_jsonl

# Frozen dataset and split settings; shared by both models.
ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "xai-org/RealworldQA"
DATASET_REVISION = "17e7f75e092e47169732462ea3cdfebe911105dd"
SOURCE_SPLIT = "test"
EXPECTED_SAMPLES = 765
SEED = 73
# Exact image-byte duplicates verified in the pinned source before inference.
IMAGE_GROUPS = ((14, 485), (43, 241), (373, 552))
OUTPUT_DIR = ROOT / "paper/data/realworldqa_external_early_stopping"


def make_folds() -> dict[str, list[int]]:
    grouped = {index: group for group in IMAGE_GROUPS for index in group}
    groups = [grouped.get(index, (index,)) for index in range(EXPECTED_SAMPLES)
              if index not in grouped or index == grouped[index][0]]
    random.Random(SEED).shuffle(groups)
    folds = {"A": [], "B": []}
    for group in groups:
        fold = "A" if len(folds["A"]) + len(group) <= 383 else "B"
        folds[fold].extend(group)
    return {fold: sorted(indices) for fold, indices in folds.items()}


def main() -> None:
    folds = make_folds()
    assert len(folds["A"]) == 383 and len(folds["B"]) == 382
    assert set(folds["A"]).isdisjoint(folds["B"])
    assert sorted(folds["A"] + folds["B"]) == list(range(EXPECTED_SAMPLES))
    assert all(set(group) <= set(folds["A"]) or set(group) <= set(folds["B"]) for group in IMAGE_GROUPS)
    hashes = {}
    for fold, indices in folds.items():
        path = OUTPUT_DIR / f"fold_{fold.lower()}.jsonl"
        write_jsonl(path, ({"source_row_index": index, "fold": fold} for index in indices))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(OUTPUT_DIR / "split.json", {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_split": SOURCE_SPLIT,
        "source_config": "default",
        "sample_count": EXPECTED_SAMPLES,
        "id_definition": "zero-based row position in the pinned official test split",
        "seed": SEED,
        "method": "source-order image groups; random.Random(seed).shuffle(groups); fill A to 383 without splitting a group; remaining B",
        "sampling": "without replacement; full coverage; no task/answer stratification; identical images grouped",
        "image_groups": IMAGE_GROUPS,
        "fold_counts": {fold: len(indices) for fold, indices in folds.items()},
        "sha256": hashes,
    })
    print(f"Frozen RealWorldQA: A=383, B=382, seed={SEED}; {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
