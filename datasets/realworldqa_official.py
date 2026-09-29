"""Prepare pinned RealWorldQA images and verbatim questions for early stopping."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

from datasets import Image, load_dataset
from PIL import Image as PILImage
from realworldqa_early_stopping_folds import DATASET_ID, DATASET_REVISION, EXPECTED_SAMPLES
from vlm_core.io import read_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper/data/realworldqa"
MANIFEST = OUTPUT_DIR / "manifest.jsonl"
SOURCE = OUTPUT_DIR / "source.json"
SOURCE_SPLIT = "test"


def main() -> None:
    source = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "config": "default",
        "split": SOURCE_SPLIT,
        "expected_samples": EXPECTED_SAMPLES,
        "prompt_field": "question (verbatim, including official answer instruction)",
        "id_definition": "zero-based pinned source row position",
        "choice_label_pattern": r"^([A-Z])[.:]\s*",
    }
    if MANIFEST.exists() and SOURCE.exists() and json.loads(SOURCE.read_text(encoding="utf-8")) == source:
        records = read_jsonl(MANIFEST)
        if (
            [int(row["index"]) for row in records] == list(range(EXPECTED_SAMPLES))
            and all(
                Path(row["messages"][0]["value"]).is_file()
                and hashlib.sha256(Path(row["messages"][0]["value"]).read_bytes()).hexdigest() == row["image_sha256"]
                for row in records
            )
        ):
            print(f"RealWorldQA already prepared: {MANIFEST}")
            return

    dataset = load_dataset(DATASET_ID, "default", split=SOURCE_SPLIT, revision=DATASET_REVISION)
    dataset = dataset.cast_column("image", Image(decode=False))
    if len(dataset) != EXPECTED_SAMPLES:
        raise ValueError(f"Expected 765 official questions, found {len(dataset)}")
    image_dir = OUTPUT_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(dataset):
        value = row["image"]
        raw = value["bytes"] if value.get("bytes") is not None else Path(value["path"]).read_bytes()
        with PILImage.open(io.BytesIO(raw)) as image:
            suffix = image.format.lower()
        image_path = image_dir / f"{index:04d}.{suffix}"
        image_path.write_bytes(raw)
        question = str(row["question"])
        choices = re.findall(r"^([A-Z])[.:]\s*", question, flags=re.MULTILINE)
        answer = str(row["answer"]).strip().upper()
        if choices and answer not in choices:
            raise ValueError(f"Case {index}: answer is absent from official choices")
        records.append({
            "index": str(index),
            "source_row_index": index,
            "question": question,
            "answer": answer,
            "choices": choices,
            "question_type": "multiple_choice" if choices else "short_answer",
            "messages": [
                {"type": "image", "value": str(image_path)},
                {"type": "text", "value": question},
            ],
            "image_sha256": hashlib.sha256(raw).hexdigest(),
        })
    write_jsonl(MANIFEST, records)
    write_json(SOURCE, source)
    print(f"Prepared {len(records)} official RealWorldQA images/questions: {MANIFEST}")


if __name__ == "__main__":
    main()
