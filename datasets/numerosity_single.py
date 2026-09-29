"""Build the canonical left-versus-right dot numerosity dataset."""

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_core.io import file_fingerprint, write_json, write_jsonl


DATASET_ID = "numerosity_single"
VERSION = "2.0.0"
SEED = 1717
OUTPUT_DIR = ROOT / "paper" / "data" / DATASET_ID
RATIOS = (2.0, 1.5, 1.25)
AREA_CONDITIONS = ("congruent", "matched", "incongruent")
SMALLER_COUNTS = {2.0: (8, 10, 12, 14, 16), 1.5: (8, 10, 12, 14, 16), 1.25: (8, 12, 16)}
SAMPLE_COUNT = 1000
QUESTION = "Which side has more dots? A. left B. right"
DOT_COLOR = (20, 20, 20)


def main() -> None:
    manifest = build_dataset(OUTPUT_DIR)
    print(OUTPUT_DIR / "manifest.json")
    print(manifest["fingerprint"])


def build_dataset(output_dir: Path) -> dict[str, Any]:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for path in image_dir.glob("*.png"):
        path.unlink()
    rng = random.Random(SEED)
    cells = [
        (ratio, area_condition, larger_side)
        for ratio in RATIOS
        for area_condition in AREA_CONDITIONS
        for larger_side in ("left", "right")
    ]
    conditions = [(*cell, "left" if repeat % 2 == 0 else "right") for cell in cells for repeat in range(55)]
    for larger_side in ("left", "right"):
        candidates = [cell for cell in cells if cell[2] == larger_side]
        rng.shuffle(candidates)
        conditions.extend((*cell, "right") for cell in candidates[:5])
    rng.shuffle(conditions)
    rows = [build_sample(index, condition, image_dir, rng) for index, condition in enumerate(conditions)]
    splits = fixed_split(rows, rng)
    validate(rows, splits)
    config = {
        "dataset_id": DATASET_ID,
        "version": VERSION,
        "seed": SEED,
        "sample_count": SAMPLE_COUNT,
        "pairing": "none",
        "ratios": list(RATIOS),
        "area_conditions": list(AREA_CONDITIONS),
        "samples_per_ratio_area_side_cell": [55, 56],
        "split": {"train": 0.8, "test": 0.2, "stratified_by": ["ratio", "area_condition", "answer"]},
    }
    write_json(output_dir / "config.json", config)
    write_jsonl(output_dir / "samples.jsonl", rows)
    write_jsonl(output_dir / "splits.jsonl", splits)
    fingerprint = file_fingerprint(
        [
            output_dir / "config.json",
            output_dir / "samples.jsonl",
            output_dir / "splits.jsonl",
            *(output_dir / row["image_path"] for row in rows),
        ]
    )
    manifest = {
        **config,
        "generator": "datasets/numerosity_single.py",
        "samples_path": "samples.jsonl",
        "splits_path": "splits.jsonl",
        "image_dir": "images",
        "fingerprint": fingerprint,
        "split_counts": dict(Counter(row["split"] for row in splits)),
        "formal_experiments": [
            "experiments/numerosity_decodability.py",
        ],
        "formal_results": [],
        "result_status": "pending_rerun",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_sample(
    index: int,
    condition: tuple[float, str, str, str],
    image_dir: Path,
    rng: random.Random,
) -> dict[str, Any]:
    ratio, area_condition, larger_side, matched_area_bias = condition
    smaller = rng.choice(SMALLER_COUNTS[ratio])
    larger = round(smaller * ratio)
    left_count, right_count = (larger, smaller) if larger_side == "left" else (smaller, larger)
    if area_condition == "matched":
        left_radius, right_radius = matched_radii(left_count, right_count, matched_area_bias)
    else:
        left_total, right_total = target_areas(left_count, right_count, area_condition)
        left_radius = math.sqrt(left_total / (math.pi * left_count))
        right_radius = math.sqrt(right_total / (math.pi * right_count))
    left_points = sample_points(rng, left_count, (24, 232, 24, 232), left_radius)
    right_points = sample_points(rng, right_count, (280, 488, 24, 232), right_radius)
    sample_id = f"numerosity_single_v2_{index:06d}"
    image = Image.new("RGB", (512, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.line((256, 16, 256, 240), fill=(180, 180, 180), width=2)
    for x, y in left_points:
        draw.ellipse((x - left_radius, y - left_radius, x + left_radius, y + left_radius), fill=DOT_COLOR)
    for x, y in right_points:
        draw.ellipse((x - right_radius, y - right_radius, x + right_radius, y + right_radius), fill=DOT_COLOR)
    image_path = image_dir / f"{sample_id}.png"
    image.save(image_path)
    return {
        "sample_id": sample_id,
        "primitive": "numerosity_individuation",
        "subtask": "nasco_dot_array",
        "difficulty": "standard",
        "image_path": f"images/{image_path.name}",
        "question": QUESTION,
        "choices": ["A", "B"],
        "answer": "A" if larger_side == "left" else "B",
        "source": "numerosity_single_v2",
        "metadata": {
            "dataset_version": VERSION,
            "scene_id": sample_id,
            "left_count": left_count,
            "right_count": right_count,
            "larger_side": larger_side,
            "ratio": ratio,
            "area_condition": area_condition,
            "matched_area_bias": matched_area_bias if area_condition == "matched" else None,
            "left_radius": round(left_radius, 6),
            "right_radius": round(right_radius, 6),
            "left_points": [list(point) for point in left_points],
            "right_points": [list(point) for point in right_points],
            "answer_value": larger_side,
        },
    }


def target_areas(left_count: int, right_count: int, condition: str) -> tuple[float, float]:
    if condition == "congruent":
        return 180 * left_count, 180 * right_count
    return 180 * right_count, 180 * left_count


def matched_radii(left_count: int, right_count: int, area_bias: str) -> tuple[float, float]:
    candidates = [4 + index * 0.25 for index in range(33)]
    target = 180 * math.sqrt(left_count * right_count)
    return min(
        (
            (left, right)
            for left in candidates
            for right in candidates
            if (left_count * dot_pixel_area(left) > right_count * dot_pixel_area(right)) == (area_bias == "left")
        ),
        key=lambda pair: (
            abs(left_count * dot_pixel_area(pair[0]) - right_count * dot_pixel_area(pair[1])),
            abs((left_count * dot_pixel_area(pair[0]) + right_count * dot_pixel_area(pair[1])) / 2 - target),
        ),
    )


@lru_cache(maxsize=None)
def dot_pixel_area(radius: float) -> int:
    image = Image.new("L", (32, 32), 0)
    ImageDraw.Draw(image).ellipse((16 - radius, 16 - radius, 16 + radius, 16 + radius), fill=255)
    return image.histogram()[255]


def sample_points(
    rng: random.Random,
    count: int,
    bounds: tuple[int, int, int, int],
    radius: float,
) -> list[tuple[int, int]]:
    x1, x2, y1, y2 = bounds
    margin = math.ceil(radius)
    points: list[tuple[int, int]] = []
    for _ in range(count * 1000):
        if len(points) == count:
            return points
        point = (rng.randint(x1 + margin, x2 - margin), rng.randint(y1 + margin, y2 - margin))
        if all(math.dist(point, other) >= 2 * math.ceil(radius) + 4 for other in points):
            points.append(point)
    raise RuntimeError(f"Could not place {count} non-overlapping dots with radius {radius:.2f}.")


def fixed_split(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, str]]:
    groups: dict[tuple[Any, ...], list[str]] = {}
    for row in rows:
        metadata = row["metadata"]
        key = (metadata["ratio"], metadata["area_condition"], row["answer"])
        groups.setdefault(key, []).append(row["sample_id"])
    test_ids = set()
    extra_by_answer = {"A": 1, "B": 1}
    for key, sample_ids in sorted(groups.items()):
        rng.shuffle(sample_ids)
        count = 11 + (1 if extra_by_answer["A" if key[2] == "A" else "B"] else 0)
        extra_by_answer["A" if key[2] == "A" else "B"] = 0
        test_ids.update(sample_ids[:count])
    return [
        {"sample_id": row["sample_id"], "split": "test" if row["sample_id"] in test_ids else "train"}
        for row in rows
    ]


def validate(rows: list[dict[str, Any]], splits: list[dict[str, str]]) -> None:
    assert len(rows) == SAMPLE_COUNT == len({row["sample_id"] for row in rows})
    assert Counter(row["answer"] for row in rows) == Counter({"A": SAMPLE_COUNT // 2, "B": SAMPLE_COUNT // 2})
    assert all("pair_id" not in row and "counterfactual_id" not in row for row in rows)
    assert {row["sample_id"] for row in rows} == {row["sample_id"] for row in splits}
    assert Counter(row["split"] for row in splits) == Counter({"train": 800, "test": 200})
    for row in rows:
        metadata = row["metadata"]
        assert row["answer"] == ("A" if metadata["left_count"] > metadata["right_count"] else "B")
        for points, radius in (
            (metadata["left_points"], metadata["left_radius"]),
            (metadata["right_points"], metadata["right_radius"]),
        ):
            assert all(
                math.dist(first, second) >= 2 * math.ceil(radius) + 3.99
                for i, first in enumerate(points)
                for second in points[i + 1 :]
            )


if __name__ == "__main__":
    main()
