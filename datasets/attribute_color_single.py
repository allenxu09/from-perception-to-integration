"""Build the canonical single-image outlined-target color dataset."""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_core.io import file_fingerprint, write_json, write_jsonl


DATASET_ID = "attribute_color_single"
VERSION = "2.0.0"
SEED = 2307
SAMPLE_COUNT = 1000
OUTPUT_DIR = ROOT / "paper" / "data" / DATASET_ID
COLORS = {
    "red": (220, 45, 45),
    "blue": (50, 105, 220),
    "green": (55, 155, 80),
    "yellow": (235, 190, 45),
}
SHAPES = ("square", "circle", "triangle", "star")
SIZES = (36, 40, 44)
OBJECT_COUNTS = (4, 5, 6, 7, 8)
OUTLINE_MARGIN = 8
JITTER = 5
LAYOUTS = {
    "grid_4x3": [(x, y) for y in (104, 256, 408) for x in (82, 198, 314, 430)],
    "grid_3x4": [(x, y) for y in (70, 194, 318, 442) for x in (96, 256, 416)],
    "staggered_4x3": [
        *( (x, 104) for x in (82, 198, 314, 430) ),
        *( (x, 256) for x in (70, 186, 302, 418) ),
        *( (x, 408) for x in (94, 210, 326, 442) ),
    ],
}
ANSWER_BY_COLOR = {color: chr(ord("A") + index) for index, color in enumerate(COLORS)}
QUESTION = (
    "Question: What color is the outlined object?\n"
    "A. red\nB. blue\nC. green\nD. yellow\n"
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
    schedules = {
        "color": balanced_schedule(tuple(COLORS), SAMPLE_COUNT, rng),
        "shape": balanced_schedule(SHAPES, SAMPLE_COUNT, rng),
        "size": balanced_schedule(SIZES, SAMPLE_COUNT, rng),
        "n_objects": balanced_schedule(OBJECT_COUNTS, SAMPLE_COUNT, rng),
        "layout": balanced_schedule(tuple(LAYOUTS), SAMPLE_COUNT, rng),
        "slot": balanced_schedule(tuple(range(12)), SAMPLE_COUNT, rng),
    }
    rows = [build_sample(index, image_dir, rng, schedules) for index in range(SAMPLE_COUNT)]
    splits = fixed_split(rows, rng)
    validate(rows, splits)
    config = {
        "dataset_id": DATASET_ID,
        "version": VERSION,
        "seed": SEED,
        "sample_count": SAMPLE_COUNT,
        "pairing": "none",
        "difficulty": "standard",
        "colors": COLORS,
        "shapes": list(SHAPES),
        "object_counts": list(OBJECT_COUNTS),
        "object_half_sizes": list(SIZES),
        "layouts": list(LAYOUTS),
        "outline_margin": OUTLINE_MARGIN,
        "jitter": JITTER,
        "split": {"train": 0.8, "test": 0.2, "stratified_by": "target_color"},
    }
    write_json(output_dir / "config.json", config)
    write_jsonl(output_dir / "samples.jsonl", rows)
    write_jsonl(output_dir / "splits.jsonl", splits)
    image_paths = [output_dir / row["image_path"] for row in rows]
    fingerprint = file_fingerprint(
        [output_dir / "config.json", output_dir / "samples.jsonl", output_dir / "splits.jsonl", *image_paths]
    )
    manifest = {
        **config,
        "generator": "datasets/attribute_color_single.py",
        "samples_path": "samples.jsonl",
        "splits_path": "splits.jsonl",
        "image_dir": "images",
        "fingerprint": fingerprint,
        "split_counts": dict(Counter(row["split"] for row in splits)),
        "formal_experiments": ["experiments/attribute_color_decodability.py"],
        "formal_results": [],
        "result_status": "pending_rerun",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def balanced_schedule(values: tuple[Any, ...], count: int, rng: random.Random) -> list[Any]:
    schedule = [values[index % len(values)] for index in range(count)]
    rng.shuffle(schedule)
    return schedule


def build_sample(
    index: int,
    image_dir: Path,
    rng: random.Random,
    schedules: dict[str, list[Any]],
) -> dict[str, Any]:
    sample_id = f"attribute_color_single_v2_{index:06d}"
    target_color = schedules["color"][index]
    target_shape = schedules["shape"][index]
    size = schedules["size"][index]
    n_objects = schedules["n_objects"][index]
    layout = schedules["layout"][index]
    target_slot = schedules["slot"][index]
    colors = balanced_scene_values(tuple(COLORS), n_objects, rng)
    shapes = balanced_scene_values(SHAPES, n_objects, rng)
    colors.remove(target_color)
    shapes.remove(target_shape)
    rng.shuffle(colors)
    rng.shuffle(shapes)
    objects = [{"color": target_color, "shape": target_shape, "outlined": True}]
    objects.extend(
        {"color": color, "shape": shape, "outlined": False}
        for color, shape in zip(colors, shapes)
    )
    slots = list(LAYOUTS[layout])
    target_center = slots.pop(target_slot)
    rng.shuffle(slots)
    centers = [target_center, *slots[: n_objects - 1]]
    centers = [(x + rng.randint(-JITTER, JITTER), y + rng.randint(-JITTER, JITTER)) for x, y in centers]
    bboxes = [(x - size, y - size, x + size, y + size) for x, y in centers]
    target_bbox = bboxes[0]
    outline_bbox = (
        target_bbox[0] - OUTLINE_MARGIN,
        target_bbox[1] - OUTLINE_MARGIN,
        target_bbox[2] + OUTLINE_MARGIN,
        target_bbox[3] + OUTLINE_MARGIN,
    )
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    for obj, bbox in zip(objects, bboxes):
        draw_shape(draw, obj["shape"], bbox, COLORS[obj["color"]])
    draw.rectangle(outline_bbox, outline=(0, 0, 0), width=5)
    image_path = image_dir / f"{sample_id}.png"
    image.save(image_path)
    color_counts = Counter(obj["color"] for obj in objects)
    shape_counts = Counter(obj["shape"] for obj in objects)
    return {
        "sample_id": sample_id,
        "primitive": "attribute",
        "subtask": DATASET_ID,
        "difficulty": "standard",
        "image_path": f"images/{image_path.name}",
        "question": QUESTION,
        "choices": ["A", "B", "C", "D"],
        "answer": ANSWER_BY_COLOR[target_color],
        "source": f"{DATASET_ID}_v2",
        "metadata": {
            "dataset_version": VERSION,
            "scene_id": sample_id,
            "target_color": target_color,
            "target_shape": target_shape,
            "target_bbox": list(target_bbox),
            "outline_bbox": list(outline_bbox),
            "target_center": list(centers[0]),
            "layout_template": layout,
            "target_slot": target_slot,
            "n_objects": n_objects,
            "object_half_size": size,
            "target_color_count": color_counts[target_color],
            "target_shape_count": shape_counts[target_shape],
            "has_exact_twin": any(
                obj["color"] == target_color and obj["shape"] == target_shape
                for obj in objects[1:]
            ),
            "objects": [
                {**obj, "bbox": list(bbox), "center": list(center)}
                for obj, bbox, center in zip(objects, bboxes, centers)
            ],
            "answer_value": target_color,
        },
    }


def balanced_scene_values(values: tuple[str, ...], count: int, rng: random.Random) -> list[str]:
    output = list(values)
    output.extend(rng.sample(values, count - len(values)))
    return output


def fixed_split(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, str]]:
    by_color: dict[str, list[str]] = {color: [] for color in COLORS}
    for row in rows:
        by_color[row["metadata"]["target_color"]].append(row["sample_id"])
    test_ids = set()
    for sample_ids in by_color.values():
        rng.shuffle(sample_ids)
        test_ids.update(sample_ids[: len(sample_ids) // 5])
    return [
        {"sample_id": row["sample_id"], "split": "test" if row["sample_id"] in test_ids else "train"}
        for row in rows
    ]


def validate(rows: list[dict[str, Any]], splits: list[dict[str, str]]) -> None:
    assert len(rows) == SAMPLE_COUNT == len({row["sample_id"] for row in rows})
    assert Counter(row["answer"] for row in rows) == Counter({answer: SAMPLE_COUNT // 4 for answer in "ABCD"})
    assert all("pair_id" not in row and "counterfactual_id" not in row for row in rows)
    assert {row["sample_id"] for row in rows} == {row["sample_id"] for row in splits}
    assert Counter(row["split"] for row in splits) == Counter({"train": 800, "test": 200})
    for row in rows:
        metadata = row["metadata"]
        assert row["answer"] == ANSWER_BY_COLOR[metadata["target_color"]]
        x1, y1, x2, y2 = metadata["target_bbox"]
        ox1, oy1, ox2, oy2 = metadata["outline_bbox"]
        assert 0 <= ox1 < x1 < x2 < ox2 < 512 and 0 <= oy1 < y1 < y2 < oy2 < 512
        boxes = [obj["bbox"] for obj in metadata["objects"]]
        for index, first in enumerate(boxes):
            for second in boxes[index + 1 :]:
                assert first[2] < second[0] or second[2] < first[0] or first[3] < second[1] or second[3] < first[1]


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, bbox: tuple[int, int, int, int], fill: tuple[int, int, int]) -> None:
    if shape == "square":
        draw.rectangle(bbox, fill=fill)
    elif shape == "circle":
        draw.ellipse(bbox, fill=fill)
    elif shape == "triangle":
        x1, y1, x2, y2 = bbox
        draw.polygon([((x1 + x2) // 2, y1), (x1, y2), (x2, y2)], fill=fill)
    else:
        import math

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
