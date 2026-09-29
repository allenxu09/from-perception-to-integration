"""Build the frozen 1,500-row MMStar dataset directly from its official source."""

from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import Image, load_dataset
from vlm_core.io import read_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "Lin-Chen/MMStar"
DATASET_REVISION = "bc98d668301da7b14f648724866e57302778ab27"
OUTPUT_DIR = ROOT / "paper/data/mmstar"
IMAGE_DIR = OUTPUT_DIR / "images"
MANIFEST = OUTPUT_DIR / "manifest.jsonl"
SOURCE = OUTPUT_DIR / "source.json"
EXPECTED_SAMPLES = 1500
OPTION_LABEL = re.compile(
    r"(?:Options:\s*|Choices:\s*|,\s*|\n)\(?([A-D])\)?(?=[.:]?\s|[.:]?\d)"
)


def main() -> None:
    expected_source = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": "val",
        "expected_samples": EXPECTED_SAMPLES,
        "prompt_field": "question (verbatim)",
    }
    if complete(expected_source):
        print(f"MMStar already prepared: {MANIFEST}")
        return

    dataset = load_dataset(
        DATASET_ID,
        "val",
        split="val",
        revision=DATASET_REVISION,
    ).cast_column("image", Image(decode=False))
    ids = sorted(int(row["index"]) for row in dataset)
    if len(dataset) != EXPECTED_SAMPLES or ids != list(range(EXPECTED_SAMPLES)):
        raise SystemExit("Official MMStar must contain stable IDs 0..1499 exactly once.")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for row in dataset:
        case = int(row["index"])
        labels = choice_labels(str(row["question"]))
        answer = str(row["answer"]).upper()
        if answer not in labels:
            raise SystemExit(f"MMStar case {case} answer {answer} is absent from its choices.")
        suffix = Path(str(row["meta_info"]["image_path"])).suffix.lower() or ".jpg"
        image_path = IMAGE_DIR / f"{case:04d}{suffix}"
        image_bytes = raw_image_bytes(row["image"])
        if not image_path.exists() or image_path.read_bytes() != image_bytes:
            temporary = image_path.with_suffix(image_path.suffix + ".tmp")
            temporary.write_bytes(image_bytes)
            temporary.replace(image_path)
        records.append({
            "index": str(case),
            "messages": [
                {"type": "image", "value": str(image_path)},
                {"type": "text", "value": str(row["question"])},
            ],
            "question": str(row["question"]),
            "answer": answer,
            "category": str(row["category"]),
            "l2_category": str(row["l2_category"]),
            "choices": labels,
        })
    write_jsonl(MANIFEST, sorted(records, key=lambda row: int(row["index"])))
    write_json(SOURCE, expected_source)
    print(f"wrote {len(records)} official MMStar rows to {MANIFEST}")


def choice_labels(question: str) -> list[str]:
    return list(dict.fromkeys(OPTION_LABEL.findall(question)))


def raw_image_bytes(value) -> bytes:
    if value.get("bytes") is not None:
        return value["bytes"]
    if value.get("path"):
        return Path(value["path"]).read_bytes()
    raise ValueError("MMStar image has neither bytes nor path.")


def complete(expected_source: dict) -> bool:
    if not MANIFEST.exists() or not SOURCE.exists():
        return False
    if json.loads(SOURCE.read_text(encoding="utf-8")) != expected_source:
        return False
    rows = read_jsonl(MANIFEST)
    return (
        len(rows) == EXPECTED_SAMPLES
        and sorted(int(row["index"]) for row in rows) == list(range(EXPECTED_SAMPLES))
        and all(Path(row["messages"][0]["value"]).is_file() for row in rows)
    )


if __name__ == "__main__":
    main()
