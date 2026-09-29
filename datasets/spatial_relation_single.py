"""Build the canonical marked-object spatial-relation dataset."""

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_core.io import file_fingerprint, write_json, write_jsonl


DATASET_ID = "spatial_relation_single"
VERSION = "2.0.0"
SEED = 3131
OUTPUT_DIR = ROOT / "paper" / "data" / DATASET_ID
RELATIONS = ("left", "right", "above", "below")
ANSWER_BY_RELATION = {relation: chr(ord("A") + index) for index, relation in enumerate(RELATIONS)}
DISTANCES = (60, 90, 120)
OBJECT_COUNTS = (4, 6, 8)
SIZES = (12, 15, 18)
SAMPLE_COUNT = 1000
OUTLINE_MARGIN = 8
COLORS = {
    "red": (220, 45, 45),
    "blue": (50, 105, 220),
    "green": (55, 155, 80),
    "yellow": (235, 190, 45),
    "purple": (145, 80, 185),
    "orange": (230, 130, 45),
}
SHAPES = ("square", "circle", "triangle", "star")
QUESTION = (
    "Question: Where is the black-outlined object relative to the gray-outlined object?\n"
    "A. left\nB. right\nC. above\nD. below\n"
    "Answer with one option only."
)


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
        (relation, distance, n_objects)
        for relation in RELATIONS
        for distance in DISTANCES
        for n_objects in OBJECT_COUNTS
    ]
    conditions = [cell for cell in cells for _ in range(27)]
    for relation in RELATIONS:
        candidates = [cell for cell in cells if cell[0] == relation]
        rng.shuffle(candidates)
        conditions.extend(candidates[:7])
    rng.shuffle(conditions)
    sizes = [SIZES[index % len(SIZES)] for index in range(SAMPLE_COUNT)]
    rng.shuffle(sizes)
    rows = [build_sample(index, condition, sizes[index], image_dir, rng) for index, condition in enumerate(conditions)]
    splits = fixed_split(rows, rng)
    validate(rows, splits)
    config = {
        "dataset_id": DATASET_ID,
        "version": VERSION,
        "seed": SEED,
        "sample_count": SAMPLE_COUNT,
        "pairing": "none",
        "relations": list(RELATIONS),
        "distances": list(DISTANCES),
        "object_counts": list(OBJECT_COUNTS),
        "object_half_sizes": list(SIZES),
        "outline_margin": OUTLINE_MARGIN,
        "split": {"train": 0.8, "test": 0.2, "stratified_by": ["relation", "distance", "n_objects"]},
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
        "generator": "datasets/spatial_relation_single.py",
        "samples_path": "samples.jsonl",
        "splits_path": "splits.jsonl",
        "image_dir": "images",
        "fingerprint": fingerprint,
        "split_counts": dict(Counter(row["split"] for row in splits)),
        "formal_experiments": ["experiments/spatial_relation_decodability.py"],
        "formal_results": [],
        "result_status": "pending_rerun",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_sample(
    index: int,
    condition: tuple[str, int, int],
    size: int,
    image_dir: Path,
    rng: random.Random,
) -> dict[str, Any]:
    relation, distance, n_objects = condition
    midpoint = (rng.randint(87, 360), rng.randint(87, 360))
    subject_center, reference_center = relation_centers(relation, midpoint, distance)
    centers = [subject_center, reference_center]
    while len(centers) < n_objects:
        centers.append(sample_free_center(rng, centers, size))
    objects = [
        {"role": "subject", "center": subject_center, "shape": rng.choice(SHAPES), "color": rng.choice(tuple(COLORS))},
        {"role": "reference", "center": reference_center, "shape": rng.choice(SHAPES), "color": rng.choice(tuple(COLORS))},
    ]
    objects.extend(
        {"role": "distractor", "center": center, "shape": rng.choice(SHAPES), "color": rng.choice(tuple(COLORS))}
        for center in centers[2:]
    )
    bboxes = [(x - size, y - size, x + size, y + size) for x, y in centers]
    sample_id = f"spatial_relation_single_v2_{index:06d}"
    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    for obj, bbox in zip(objects, bboxes):
        draw_shape(draw, obj["shape"], bbox, COLORS[obj["color"]])
    outlines = []
    for index_in_scene, color in ((0, (0, 0, 0)), (1, (130, 130, 130))):
        x1, y1, x2, y2 = bboxes[index_in_scene]
        outline = (x1 - OUTLINE_MARGIN, y1 - OUTLINE_MARGIN, x2 + OUTLINE_MARGIN, y2 + OUTLINE_MARGIN)
        draw.rectangle(outline, outline=color, width=5)
        outlines.append(outline)
    image_path = image_dir / f"{sample_id}.png"
    image.save(image_path)
    return {
        "sample_id": sample_id,
        "primitive": "spatial_relation",
        "subtask": DATASET_ID,
        "difficulty": "standard",
        "image_path": f"images/{image_path.name}",
        "question": QUESTION,
        "choices": ["A", "B", "C", "D"],
        "answer": ANSWER_BY_RELATION[relation],
        "source": "spatial_relation_single_v2",
        "metadata": {
            "dataset_version": VERSION,
            "scene_id": sample_id,
            "true_relation_label": relation,
            "distance": distance,
            "n_objects": n_objects,
            "object_half_size": size,
            "midpoint": list(midpoint),
            "subject_bbox": list(bboxes[0]),
            "reference_bbox": list(bboxes[1]),
            "subject_outline_bbox": list(outlines[0]),
            "reference_outline_bbox": list(outlines[1]),
            "objects": [{**obj, "center": list(obj["center"]), "bbox": list(bbox)} for obj, bbox in zip(objects, bboxes)],
            "answer_value": relation,
        },
    }


def relation_centers(relation: str, midpoint: tuple[int, int], distance: int) -> tuple[tuple[int, int], tuple[int, int]]:
    x, y = midpoint
    half = distance // 2
    if relation == "left":
        return (x - half, y), (x + half, y)
    if relation == "right":
        return (x + half, y), (x - half, y)
    if relation == "above":
        return (x, y - half), (x, y + half)
    return (x, y + half), (x, y - half)


def sample_free_center(rng: random.Random, centers: list[tuple[int, int]], size: int) -> tuple[int, int]:
    separation = 2 * size + OUTLINE_MARGIN + 4
    for _ in range(2000):
        center = (rng.randint(size + 10, 438 - size), rng.randint(size + 10, 438 - size))
        if all(abs(center[0] - other[0]) >= separation or abs(center[1] - other[1]) >= separation for other in centers):
            return center
    raise RuntimeError("Could not place spatial distractor without overlap.")


def fixed_split(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, str]]:
    groups: dict[tuple[Any, ...], list[str]] = {}
    for row in rows:
        metadata = row["metadata"]
        key = (metadata["true_relation_label"], metadata["distance"], metadata["n_objects"])
        groups.setdefault(key, []).append(row["sample_id"])
    test_ids = set()
    extras_by_relation = {relation: 5 for relation in RELATIONS}
    for key, sample_ids in sorted(groups.items()):
        rng.shuffle(sample_ids)
        count = 5 + (1 if extras_by_relation[key[0]] else 0)
        extras_by_relation[key[0]] = max(0, extras_by_relation[key[0]] - 1)
        test_ids.update(sample_ids[:count])
    return [
        {"sample_id": row["sample_id"], "split": "test" if row["sample_id"] in test_ids else "train"}
        for row in rows
    ]


def validate(rows: list[dict[str, Any]], splits: list[dict[str, str]]) -> None:
    assert len(rows) == SAMPLE_COUNT == len({row["sample_id"] for row in rows})
    assert Counter(row["answer"] for row in rows) == Counter({answer: SAMPLE_COUNT // 4 for answer in "ABCD"})
    assert all("pair_id" not in row and "counterfactual_id" not in row for row in rows)
    assert Counter(row["split"] for row in splits) == Counter({"train": 800, "test": 200})
    for row in rows:
        metadata = row["metadata"]
        assert row["answer"] == ANSWER_BY_RELATION[metadata["true_relation_label"]]
        boxes = [obj["bbox"] for obj in metadata["objects"]]
        for index, first in enumerate(boxes):
            for second in boxes[index + 1 :]:
                assert first[2] < second[0] or second[2] < first[0] or first[3] < second[1] or second[3] < first[1]
        for outline in (metadata["subject_outline_bbox"], metadata["reference_outline_bbox"]):
            assert 0 <= min(outline) and max(outline) < 448


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, bbox: tuple[int, int, int, int], fill: tuple[int, int, int]) -> None:
    if shape == "square":
        draw.rectangle(bbox, fill=fill)
    elif shape == "circle":
        draw.ellipse(bbox, fill=fill)
    elif shape == "triangle":
        x1, y1, x2, y2 = bbox
        draw.polygon([((x1 + x2) // 2, y1), (x1, y2), (x2, y2)], fill=fill)
    else:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        outer, inner = (x2 - x1) / 2, (x2 - x1) * 0.225
        draw.polygon(
            [
                (
                    cx + (outer if i % 2 == 0 else inner) * math.cos(-math.pi / 2 + i * math.pi / 5),
                    cy + (outer if i % 2 == 0 else inner) * math.sin(-math.pi / 2 + i * math.pi / 5),
                )
                for i in range(10)
            ],
            fill=fill,
        )


if __name__ == "__main__":
    main()
