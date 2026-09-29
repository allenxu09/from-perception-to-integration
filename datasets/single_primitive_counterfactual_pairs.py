"""Build true clean/corrupt pairs for formal Counterfactual activation patching patching."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from vlm_core.io import write_json, write_jsonl


COLORS = {
    "red": (220, 45, 45),
    "blue": (50, 105, 220),
    "green": (55, 155, 80),
    "yellow": (235, 190, 45),
}
SHAPES = ("square", "circle", "triangle", "star")
COLOR_OPTIONS = ("red", "blue", "green", "yellow")
COLOR_ANSWER = {color: chr(ord("A") + index) for index, color in enumerate(COLOR_OPTIONS)}
RELATIONS = ("left", "right", "above", "below")
RELATION_ANSWER = {label: chr(ord("A") + index) for index, label in enumerate(RELATIONS)}

OUTPUT_DIR = Path("data/single_primitive_counterfactual_pairs")
PAIRS_PER_PRIMITIVE = 300
SEED = 41


def main() -> None:
    out_dir = OUTPUT_DIR
    rng = random.Random(SEED)
    samples, pairs = [], []
    builders = [
        ("attribute", build_attribute_pair),
        ("numerosity", build_numerosity_pair),
        ("spatial", build_spatial_pair),
        ("occlusion", build_occlusion_pair),
    ]
    for primitive, builder in builders:
        image_dir = out_dir / primitive / "images"
        for index in range(PAIRS_PER_PRIMITIVE):
            pair_samples, pair = builder(index, image_dir, rng)
            samples.extend(pair_samples)
            pairs.append(pair)

    write_jsonl(out_dir / "samples.jsonl", samples)
    write_jsonl(out_dir / "pairs.jsonl", pairs)
    write_json(
        out_dir / "manifest.json",
        {
            "dataset_id": "single_primitive_counterfactual_pairs_v1",
            "template_id": "single_primitive_counterfactual_pairs_v1",
            "generator": "datasets/single_primitive_counterfactual_pairs.py",
            "output_dir": str(out_dir).replace("\\", "/"),
            "pairs_per_primitive": PAIRS_PER_PRIMITIVE,
            "sample_count": len(samples),
            "pair_count": len(pairs),
            "samples_path": str(out_dir / "samples.jsonl"),
            "pairs_path": str(out_dir / "pairs.jsonl"),
        },
    )
    print(out_dir / "samples.jsonl")


def build_attribute_pair(index: int, image_dir: Path, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_id = f"attribute_pair_{index:06d}"
    clean_color = COLOR_OPTIONS[index % len(COLOR_OPTIONS)]
    corrupt_color = COLOR_OPTIONS[(index + 1) % len(COLOR_OPTIONS)]
    target_shape = SHAPES[(index // len(COLOR_OPTIONS)) % len(SHAPES)]
    positions = grid_positions()
    rng.shuffle(positions)
    objects = [
        {"color": clean_color, "shape": target_shape, "outlined": True, "center": positions[0]},
        {"color": COLOR_OPTIONS[(index + 2) % len(COLOR_OPTIONS)], "shape": target_shape, "outlined": False, "center": positions[1]},
        {"color": clean_color, "shape": SHAPES[(SHAPES.index(target_shape) + 1) % len(SHAPES)], "outlined": False, "center": positions[2]},
        {"color": COLOR_OPTIONS[(index + 3) % len(COLOR_OPTIONS)], "shape": SHAPES[(SHAPES.index(target_shape) + 2) % len(SHAPES)], "outlined": False, "center": positions[3]},
        {"color": COLOR_OPTIONS[(index + 1) % len(COLOR_OPTIONS)], "shape": SHAPES[(SHAPES.index(target_shape) + 3) % len(SHAPES)], "outlined": False, "center": positions[4]},
        {"color": COLOR_OPTIONS[(index + 2) % len(COLOR_OPTIONS)], "shape": SHAPES[(SHAPES.index(target_shape) + 2) % len(SHAPES)], "outlined": False, "center": positions[5]},
    ]
    question = "Question: What color is the outlined object?\nA. red\nB. blue\nC. green\nD. yellow\nAnswer with one option only."

    def make(which: str, color: str) -> dict[str, Any]:
        sample_objects = [dict(obj) for obj in objects]
        sample_objects[0]["color"] = color
        image_path = image_dir / f"{pair_id}_{which}.png"
        bboxes = draw_objects(image_path, sample_objects, 44)
        return sample_row(
            pair_id,
            which,
            "attribute",
            "attribute_outlined_target_cf_v1",
            image_path,
            question,
            ["A", "B", "C", "D"],
            COLOR_ANSWER[color],
            {
                "target_color": color,
                "target_shape": target_shape,
                "target_bbox": list(bboxes[0]),
                "target_position": position_name(bboxes[0]),
                "changed_factor": "outlined_target_color",
                "template_id": "attribute_outlined_target_cf_v1",
            },
        )

    clean = make("clean", clean_color)
    corrupt = make("corrupt", corrupt_color)
    return [clean, corrupt], pair_row(pair_id, clean, corrupt, "outlined_target_color")


def build_numerosity_pair(index: int, image_dir: Path, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_id = f"numerosity_pair_{index:06d}"
    larger = 6 + index % 4
    smaller = 3 + index % 3
    left_layout = dot_layout(larger, (128, 128), rng)
    right_layout = dot_layout(smaller, (384, 128), rng)
    question = "Question: Which side has more dots?\nA. left\nB. right\nAnswer with one option only."

    clean_path = image_dir / f"{pair_id}_clean.png"
    corrupt_path = image_dir / f"{pair_id}_corrupt.png"
    draw_dots(clean_path, left_layout, right_layout)
    draw_dots(corrupt_path, [(x + 256, y) for x, y in left_layout], [(x - 256, y) for x, y in right_layout])
    clean = sample_row(pair_id, "clean", "numerosity_individuation", "nasco_dot_array_cf_v1", clean_path, question, ["A", "B"], "A", {
        "left_count": larger,
        "right_count": smaller,
        "larger_count": larger,
        "larger_side": "left",
        "changed_factor": "larger_side",
        "template_id": "nasco_dot_array_cf_v1",
    })
    corrupt = sample_row(pair_id, "corrupt", "numerosity_individuation", "nasco_dot_array_cf_v1", corrupt_path, question, ["A", "B"], "B", {
        "left_count": smaller,
        "right_count": larger,
        "larger_count": larger,
        "larger_side": "right",
        "changed_factor": "larger_side",
        "template_id": "nasco_dot_array_cf_v1",
    })
    return [clean, corrupt], pair_row(pair_id, clean, corrupt, "larger_side")


def build_spatial_pair(index: int, image_dir: Path, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_id = f"spatial_pair_{index:06d}"
    relation, corrupt_relation = (("left", "right"), ("above", "below"))[index % 2]
    reference = (224, 224)
    distance = 132
    subject_clean = relation_center(reference, relation, distance)
    subject_corrupt = relation_center(reference, corrupt_relation, distance)
    distractors = [
        {"center": center, "shape": rng.choice(SHAPES), "color": rng.choice(COLOR_OPTIONS), "outlined": False}
        for center in [(88, 88), (360, 88), (88, 360), (360, 360)]
    ]
    question = (
        "Question: Where is the black-marked object relative to the gray-marked object?\n"
        "A. left\nB. right\nC. above\nD. below\nAnswer with one option only."
    )

    def make(which: str, label: str, subject: tuple[int, int]) -> dict[str, Any]:
        objects = [
            {"center": subject, "shape": "square", "color": "red", "outlined": True, "outline": "black"},
            {"center": reference, "shape": "circle", "color": "blue", "outlined": True, "outline": "gray"},
            *distractors,
        ]
        image_path = image_dir / f"{pair_id}_{which}.png"
        bboxes = draw_objects(image_path, objects, 30)
        return sample_row(pair_id, which, "spatial_relation", "spatial_marked_relation_cf_v1", image_path, question, ["A", "B", "C", "D"], RELATION_ANSWER[label], {
            "true_relation_label": label,
            "subject_bbox": list(bboxes[0]),
            "reference_bbox": list(bboxes[1]),
            "changed_factor": "marked_object_relation",
            "template_id": "spatial_marked_relation_cf_v1",
        })

    clean = make("clean", relation, subject_clean)
    corrupt = make("corrupt", corrupt_relation, subject_corrupt)
    return [clean, corrupt], pair_row(pair_id, clean, corrupt, "marked_object_relation")


def build_occlusion_pair(index: int, image_dir: Path, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_id = f"occlusion_pair_{index:06d}"
    question = (
        "Question: Do the visible fragments belong to one continuous object behind the occluder?\n"
        "A. one object\nB. two objects\nAnswer with one option only."
    )

    def make(which: str, label: str) -> dict[str, Any]:
        image_path = image_dir / f"{pair_id}_{which}.png"
        metadata = draw_occlusion(image_path, label, rng)
        metadata.update({
            "object_unity_label": label,
            "visible_part_count": 2,
            "whole_object_count": 1 if label == "one" else 2,
            "changed_factor": "fragment_alignment",
            "template_id": "occlusion_fragment_alignment_cf_v1",
        })
        return sample_row(pair_id, which, "occlusion_object_unity", "occlusion_fragment_alignment_cf_v1", image_path, question, ["A", "B"], "A" if label == "one" else "B", metadata)

    clean = make("clean", "one")
    corrupt = make("corrupt", "two")
    return [clean, corrupt], pair_row(pair_id, clean, corrupt, "fragment_alignment")


def sample_row(
    pair_id: str,
    which: str,
    primitive: str,
    subtask: str,
    image_path: Path,
    question: str,
    choices: list[str],
    answer: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = {**metadata, "pair_id": pair_id, "clean_or_corrupt": which}
    return {
        "sample_id": f"{pair_id}_{which}",
        "pair_id": pair_id,
        "counterfactual_id": pair_id,
        "primitive": primitive,
        "subtask": subtask,
        "difficulty": "formal",
        "image_path": str(image_path).replace("\\", "/"),
        "question": question,
        "choices": choices,
        "answer": answer,
        "source": "synthetic_single_primitive_counterfactual_pairs_v1",
        "metadata": metadata,
    }


def pair_row(pair_id: str, clean: dict[str, Any], corrupt: dict[str, Any], changed_factor: str) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "primitive": clean["primitive"],
        "subtype": clean["subtask"],
        "source_dataset": clean["source"],
        "clean_image": clean["image_path"],
        "corrupt_image": corrupt["image_path"],
        "prompt": clean["question"],
        "clean_answer": clean["answer"],
        "corrupt_answer": corrupt["answer"],
        "changed_factor": changed_factor,
        "metadata": {"clean": clean["metadata"], "corrupt": corrupt["metadata"]},
    }


def grid_positions() -> list[tuple[int, int]]:
    return [(96, 112), (224, 112), (352, 112), (96, 248), (224, 248), (352, 248)]


def draw_objects(image_path: Path, objects: list[dict[str, Any]], size: int) -> list[tuple[int, int, int, int]]:
    image = Image.new("RGB", (448, 360), "white")
    draw = ImageDraw.Draw(image)
    bboxes = []
    for obj in objects:
        cx, cy = obj["center"]
        bbox = (cx - size, cy - size, cx + size, cy + size)
        draw_shape(draw, str(obj["shape"]), bbox, COLORS[str(obj["color"])])
        if obj.get("outlined"):
            outline = (0, 0, 0) if obj.get("outline", "black") == "black" else (130, 130, 130)
            draw.rectangle((bbox[0] - 6, bbox[1] - 6, bbox[2] + 6, bbox[3] + 6), outline=outline, width=5)
        bboxes.append(bbox)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    return bboxes


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, bbox: tuple[int, int, int, int], fill: tuple[int, int, int]) -> None:
    if shape == "square":
        draw.rectangle(bbox, fill=fill)
    elif shape == "circle":
        draw.ellipse(bbox, fill=fill)
    elif shape == "triangle":
        x1, y1, x2, y2 = bbox
        draw.polygon([((x1 + x2) // 2, y1), (x1, y2), (x2, y2)], fill=fill)
    elif shape == "star":
        draw.polygon(star_points(bbox), fill=fill)


def star_points(bbox: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    outer = (x2 - x1) / 2
    inner = outer * 0.45
    return [
        (
            cx + (outer if i % 2 == 0 else inner) * math.cos(-math.pi / 2 + i * math.pi / 5),
            cy + (outer if i % 2 == 0 else inner) * math.sin(-math.pi / 2 + i * math.pi / 5),
        )
        for i in range(10)
    ]


def position_name(bbox: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return f"{'top' if cy < 180 else 'bottom'}_{'left' if cx < 150 else 'center' if cx < 300 else 'right'}"


def dot_layout(n: int, center: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    points = []
    while len(points) < n:
        x = center[0] + rng.randint(-78, 78)
        y = center[1] + rng.randint(-78, 78)
        if all((x - px) ** 2 + (y - py) ** 2 > 18**2 for px, py in points):
            points.append((x, y))
    return points


def draw_dots(image_path: Path, left: list[tuple[int, int]], right: list[tuple[int, int]]) -> None:
    image = Image.new("RGB", (512, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.line((256, 20, 256, 236), fill=(180, 180, 180), width=2)
    for x, y in left + right:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(35, 35, 35))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)


def relation_center(reference: tuple[int, int], relation: str, distance: int) -> tuple[int, int]:
    x, y = reference
    if relation == "left":
        return x - distance, y
    if relation == "right":
        return x + distance, y
    if relation == "above":
        return x, y - distance
    return x, y + distance


def draw_occlusion(image_path: Path, label: str, rng: random.Random) -> dict[str, Any]:
    image = Image.new("RGB", (448, 360), "white")
    draw = ImageDraw.Draw(image)
    y = 180
    color = (70, 130, 210)
    height = 32
    offset = 0 if label == "one" else 34
    left = (80, y - height // 2, 190, y + height // 2)
    right = (258, y + offset - height // 2, 368, y + offset + height // 2)
    occluder = (190, 92, 258, 268)
    draw.rounded_rectangle(left, radius=height // 2, fill=color)
    draw.rounded_rectangle(right, radius=height // 2, fill=color)
    draw.rectangle(occluder, fill=(205, 205, 205), outline=(90, 90, 90), width=2)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    return {
        "left_fragment_bbox": list(left),
        "right_fragment_bbox": list(right),
        "occluder_bbox": list(occluder),
        "alignment_offset": offset,
        "orientation_difference": 0,
    }


if __name__ == "__main__":
    main()
