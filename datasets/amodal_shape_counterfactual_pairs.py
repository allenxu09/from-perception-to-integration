"""Build the canonical diverse matched-pair amodal shape dataset."""

from __future__ import annotations

import math
import random
import sys
from heapq import heapify, heappop, heappush
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_core.io import file_fingerprint, write_json, write_jsonl


DATASET_ID = "occlusion_v2_amodal_fixed_1000"
VERSION = "2.0.0"
SEED = 4191
OUTPUT_DIR = ROOT / "paper" / "data" / DATASET_ID
CANVAS = (512, 384)
SHAPES = ("circle", "triangle", "rectangle", "star", "L-shape", "irregular polygon")
LETTERS = ("A", "B", "C", "D")
COLORS = {
    "red": (210, 70, 60),
    "blue": (55, 105, 210),
    "green": (55, 155, 85),
    "orange": (225, 130, 45),
    "purple": (145, 80, 185),
    "yellow": (225, 180, 35),
}
OCCLUDERS = (
    "horizontal_rectangle",
    "vertical_rectangle",
    "diagonal_rectangle",
    "ellipse",
    "polygon",
    "double_rectangle",
)
PAIR_COUNT = 500
GRAY = (195, 195, 195)


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
    shape_pairs = balanced_pairs(SHAPES, [167, 167, 167, 167, 166, 166], rng)
    letter_pairs = balanced_edges(LETTERS)
    rng.shuffle(letter_pairs)
    colors = balanced_values(tuple(COLORS), PAIR_COUNT, rng)
    occluder_types = balanced_values(OCCLUDERS, PAIR_COUNT, rng)
    target_ratios = [0.28 + 0.34 * index / (PAIR_COUNT - 1) for index in range(PAIR_COUNT)]
    rng.shuffle(target_ratios)
    rows: list[dict[str, Any]] = []
    for pair_index, values in enumerate(zip(shape_pairs, letter_pairs, colors, occluder_types, target_ratios)):
        rows.extend(build_pair(pair_index, *values, image_dir, rng))
    pair_type = {row["pair_id"]: row["metadata"]["occluder_type"] for row in rows}
    test_pairs = set()
    for index, occluder_type in enumerate(OCCLUDERS):
        candidates = [pair_id for pair_id, value in pair_type.items() if value == occluder_type]
        rng.shuffle(candidates)
        test_pairs.update(candidates[: 17 if index < 4 else 16])
    splits = [
        {"sample_id": row["sample_id"], "pair_id": row["pair_id"], "split": "test" if row["pair_id"] in test_pairs else "train"}
        for row in rows
    ]
    validate(rows, splits)
    config = {
        "dataset_id": DATASET_ID,
        "version": VERSION,
        "seed": SEED,
        "sample_count": len(rows),
        "pair_count": PAIR_COUNT,
        "pairing": "matched_counterfactual",
        "changed_factor": "complete_shape",
        "shapes": list(SHAPES),
        "colors": list(COLORS),
        "occluder_types": list(OCCLUDERS),
        "occlusion_ratio_range": [0.25, 0.65],
        "minimum_visible_components": 2,
        "split": {"train_pairs": 400, "test_pairs": 100, "grouped_by": "pair_id", "stratified_by": "occluder_type"},
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
        "generator": "datasets/amodal_shape_counterfactual_pairs.py",
        "samples_path": "samples.jsonl",
        "splits_path": "splits.jsonl",
        "image_dir": "images",
        "fingerprint": fingerprint,
        "formal_experiments": [
            "experiments/counterfactual_activation_patching.py",
            "experiments/counterfactual_gradient_attribution.py",
            "experiments/pathway_availability.py",
            "experiments/rescue_set_decodability.py",
        ],
        "formal_results": [],
        "result_status": "pending_rerun",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def balanced_pairs(values: tuple[str, ...], counts: list[int], rng: random.Random) -> list[tuple[str, str]]:
    heap = [(-count, value) for value, count in zip(values, counts)]
    heapify(heap)
    pairs = []
    while heap:
        left_count, left = heappop(heap)
        right_count, right = heappop(heap)
        pairs.append((left, right))
        if left_count + 1:
            heappush(heap, (left_count + 1, left))
        if right_count + 1:
            heappush(heap, (right_count + 1, right))
    rng.shuffle(pairs)
    return pairs


def balanced_edges(values: tuple[str, ...]) -> list[tuple[str, str]]:
    counts = (84, 83, 83, 83, 83, 84)
    edges = [(values[i], values[j]) for i in range(4) for j in range(i + 1, 4)]
    return [edge for edge, count in zip(edges, counts) for _ in range(count)]


def balanced_values(values: tuple[str, ...], count: int, rng: random.Random) -> list[str]:
    result = [values[index % len(values)] for index in range(count)]
    rng.shuffle(result)
    return result


def build_pair(
    pair_index: int,
    shape_pair: tuple[str, str],
    letter_pair: tuple[str, str],
    color_name: str,
    occluder_type: str,
    target_ratio: float,
    image_dir: Path,
    rng: random.Random,
) -> list[dict[str, Any]]:
    last_error = None
    for _ in range(50):
        size = rng.randint(145, 215)
        center = (rng.randint(130, 382), rng.randint(120, 264))
        rotation = rng.randint(0, 359)
        masks = {shape: shape_mask(shape, center, size, rotation, rng) for shape in shape_pair}
        try:
            occluder_mask, ratios, components, occluder_params = matched_occluder(
                masks, center, size, occluder_type, target_ratio, rng
            )
            break
        except RuntimeError as error:
            last_error = error
            continue
    else:
        raise RuntimeError(
            f"Could not build matched pair {shape_pair} with {occluder_type} at target ratio {target_ratio:.3f}: {last_error}"
        )
    first_letter, second_letter = letter_pair
    if rng.random() < 0.5:
        first_letter, second_letter = second_letter, first_letter
    order: list[str | None] = [None] * 4
    order[LETTERS.index(first_letter)] = shape_pair[0]
    order[LETTERS.index(second_letter)] = shape_pair[1]
    foils = [shape for shape in SHAPES if shape not in shape_pair]
    rng.shuffle(foils)
    for index in range(4):
        if order[index] is None:
            order[index] = foils.pop()
    candidate_order = [str(value) for value in order]
    pair_id = f"amodal_pair_v2_{pair_index:04d}"
    pair_shapes = list(shape_pair)
    if rng.random() < 0.5:
        pair_shapes.reverse()
    pair_rows = []
    for side, shape in zip(("clean", "corrupt"), pair_shapes):
        sample_id = f"{pair_id}_{side}"
        image = Image.new("RGB", CANVAS, "white")
        image.paste(COLORS[color_name], mask=masks[shape])
        image.paste(GRAY, mask=occluder_mask)
        image_path = image_dir / f"{sample_id}.png"
        image.save(image_path)
        pair_rows.append(
            {
                "sample_id": sample_id,
                "pair_id": pair_id,
                "counterfactual_id": pair_id,
                "primitive": "occlusion_amodal_completion",
                "subtask": "amodal_completion_v2",
                "difficulty": occlusion_bin(ratios[shape]),
                "image_path": f"images/{image_path.name}",
                "question": option_prompt(color_name, candidate_order),
                "choices": list(LETTERS),
                "answer": LETTERS[candidate_order.index(shape)],
                "source": "amodal_completion_v2",
                "metadata": {
                    "dataset_version": VERSION,
                    "clean_or_corrupt": side,
                    "changed_factor": "complete_shape",
                    "complete_shape": shape,
                    "complete_shape_label": shape,
                    "candidate_order": candidate_order,
                    "answer_value": shape,
                    "target_color": color_name,
                    "target_size": size,
                    "target_rotation_degrees": rotation,
                    "target_center": list(center),
                    "target_bbox_actual": list(masks[shape].getbbox() or ()),
                    "occluder_type": occluder_type,
                    "occluder_params": occluder_params,
                    "occlusion_ratio_target": round(target_ratio, 4),
                    "occlusion_ratio_actual": round(ratios[shape], 4),
                    "visible_component_count": components[shape],
                    "template_id": "amodal_completion_v2",
                },
            }
        )
    return pair_rows


def shape_mask(shape: str, center: tuple[int, int], size: int, rotation: int, rng: random.Random) -> Image.Image:
    mask = Image.new("L", CANVAS, 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = center
    half = size // 2
    box = (cx - half, cy - half, cx + half, cy + half)
    if shape == "circle":
        draw.ellipse(box, fill=255)
    elif shape == "rectangle":
        inset = rng.randint(0, size // 8)
        draw.rectangle((box[0] + inset, box[1], box[2] - inset, box[3]), fill=255)
    elif shape == "triangle":
        draw.polygon([(cx, box[1]), (box[0], box[3]), (box[2], box[3])], fill=255)
    elif shape == "star":
        draw.polygon(radial_points(center, half, half * rng.uniform(0.38, 0.52), 5), fill=255)
    elif shape == "L-shape":
        arm = rng.randint(size // 4, size // 2)
        draw.polygon(
            [(box[0], box[1]), (box[0] + arm, box[1]), (box[0] + arm, box[3] - arm),
             (box[2], box[3] - arm), (box[2], box[3]), (box[0], box[3])],
            fill=255,
        )
    else:
        vertices = rng.randint(6, 9)
        radii = [half * rng.uniform(0.62, 1.0) for _ in range(vertices)]
        draw.polygon(
            [
                (cx + radius * math.cos(-math.pi / 2 + 2 * math.pi * index / vertices),
                 cy + radius * math.sin(-math.pi / 2 + 2 * math.pi * index / vertices))
                for index, radius in enumerate(radii)
            ],
            fill=255,
        )
    return mask.rotate(rotation, resample=Image.Resampling.NEAREST, center=center)


def radial_points(center: tuple[int, int], outer: float, inner: float, tips: int) -> list[tuple[float, float]]:
    cx, cy = center
    return [
        (cx + (outer if i % 2 == 0 else inner) * math.cos(-math.pi / 2 + i * math.pi / tips),
         cy + (outer if i % 2 == 0 else inner) * math.sin(-math.pi / 2 + i * math.pi / tips))
        for i in range(tips * 2)
    ]


def matched_occluder(
    masks: dict[str, Image.Image],
    center: tuple[int, int],
    size: int,
    occluder_type: str,
    target_ratio: float,
    rng: random.Random,
) -> tuple[Image.Image, dict[str, float], dict[str, int], dict[str, Any]]:
    arrays = {shape: np.asarray(mask) > 0 for shape, mask in masks.items()}
    totals = {shape: int(array.sum()) for shape, array in arrays.items()}
    best = None
    ratio_passes = 0
    component_passes = 0
    for _ in range(400):
        occluder, params = occluder_mask(occluder_type, center, size, target_ratio, rng)
        occ = np.asarray(occluder) > 0
        ratios = {shape: float(np.logical_and(array, occ).sum() / totals[shape]) for shape, array in arrays.items()}
        score = max(abs(value - target_ratio) for value in ratios.values())
        score += abs(next(iter(ratios.values())) - list(ratios.values())[1])
        if best is None or score < best[0]:
            best = (score, occluder, ratios, params)
        if all(0.25 <= value <= 0.65 for value in ratios.values()):
            ratio_passes += 1
            components = {shape: visible_components(array, occ) for shape, array in arrays.items()}
            if min(components.values()) >= 2:
                component_passes += 1
                return occluder, ratios, components, params
    raise RuntimeError(
        f"Could not place {occluder_type} for {tuple(masks)}; target={target_ratio:.3f}, "
        f"best_score={best[0] if best else None}, best_ratios={best[2] if best else None}, "
        f"ratio_passes={ratio_passes}, component_passes={component_passes}"
    )


def occluder_mask(
    kind: str, center: tuple[int, int], size: int, target_ratio: float, rng: random.Random
) -> tuple[Image.Image, dict[str, Any]]:
    cx, cy = center
    thickness = max(18, int(size * target_ratio * rng.uniform(0.50, 1.12)))
    if kind == "horizontal_rectangle":
        angle = rng.uniform(-8, 8)
    elif kind == "vertical_rectangle":
        angle = rng.uniform(82, 98)
    elif kind in ("diagonal_rectangle", "polygon"):
        angle = rng.choice((rng.uniform(22, 68), rng.uniform(112, 158)))
    else:
        angle = rng.uniform(0, 180)
    offset = rng.uniform(-0.10 * size, 0.10 * size)
    normal = (-math.sin(math.radians(angle)), math.cos(math.radians(angle)))
    shifted = (cx + normal[0] * offset, cy + normal[1] * offset)
    length = size * 1.65
    mask = Image.new("L", CANVAS, 0)
    draw = ImageDraw.Draw(mask)
    if kind == "ellipse":
        base = Image.new("L", CANVAS, 0)
        bx, by = shifted
        ImageDraw.Draw(base).ellipse(
            (bx - length / 2, by - thickness / 2, bx + length / 2, by + thickness / 2), fill=255
        )
        mask = base.rotate(angle, resample=Image.Resampling.NEAREST, center=shifted)
    elif kind == "double_rectangle":
        gap = rng.uniform(0.16, 0.28) * size
        for sign in (-1, 1):
            band_center = (shifted[0] + normal[0] * gap * sign, shifted[1] + normal[1] * gap * sign)
            draw.polygon(band_points(band_center, length, thickness * 0.55, angle), fill=255)
    elif kind == "polygon":
        points = band_points(shifted, length, thickness, angle)
        tangent = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        jitter = rng.uniform(-0.12, 0.12) * thickness
        points[1] = (points[1][0] + tangent[0] * jitter, points[1][1] + tangent[1] * jitter)
        points[3] = (points[3][0] - tangent[0] * jitter, points[3][1] - tangent[1] * jitter)
        draw.polygon(points, fill=255)
    else:
        draw.polygon(band_points(shifted, length, thickness, angle), fill=255)
    return mask, {
        "angle_degrees": round(angle, 2),
        "center": [round(shifted[0], 2), round(shifted[1], 2)],
        "length": round(length, 2),
        "thickness": round(thickness, 2),
    }


def band_points(center: tuple[float, float], length: float, thickness: float, angle: float) -> list[tuple[float, float]]:
    cx, cy = center
    tangent = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
    normal = (-tangent[1], tangent[0])
    return [
        (cx + tangent[0] * length * sx / 2 + normal[0] * thickness * sy / 2,
         cy + tangent[1] * length * sx / 2 + normal[1] * thickness * sy / 2)
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]


def visible_components(target: np.ndarray, occluder: np.ndarray) -> int:
    labels, count = label(np.logical_and(target, ~occluder), structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(labels.ravel())[1:]
    return int(np.sum(sizes >= 100)) if count else 0


def occlusion_bin(ratio: float) -> str:
    if ratio < 0.39:
        return "low"
    if ratio < 0.53:
        return "medium"
    return "high"


def option_prompt(color: str, order: list[str]) -> str:
    options = " ".join(f"{letter}. {shape}" for letter, shape in zip(LETTERS, order))
    return f"What is the complete {color} shape behind the gray occluder? {options} Answer with one option only."


def validate(rows: list[dict[str, Any]], splits: list[dict[str, str]]) -> None:
    assert len(rows) == 1000 and len({row["sample_id"] for row in rows}) == 1000
    assert Counter(row["answer"] for row in rows) == Counter({letter: 250 for letter in LETTERS})
    assert sorted(Counter(row["metadata"]["complete_shape"] for row in rows).values()) == [166, 166, 167, 167, 167, 167]
    assert Counter(row["split"] for row in splits) == Counter({"train": 800, "test": 200})
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["pair_id"], []).append(row)
        metadata = row["metadata"]
        assert row["answer"] == LETTERS[metadata["candidate_order"].index(metadata["complete_shape"])]
        assert 0.25 <= metadata["occlusion_ratio_actual"] <= 0.65
        assert metadata["visible_component_count"] >= 2
    split_by_id = {row["sample_id"]: row["split"] for row in splits}
    assert len(groups) == PAIR_COUNT
    for members in groups.values():
        assert len(members) == 2 and members[0]["answer"] != members[1]["answer"]
        for field in (
            "candidate_order", "target_color", "target_size", "target_rotation_degrees", "target_center",
            "occluder_type", "occluder_params", "occlusion_ratio_target",
        ):
            assert members[0]["metadata"][field] == members[1]["metadata"][field]
        assert split_by_id[members[0]["sample_id"]] == split_by_id[members[1]["sample_id"]]


if __name__ == "__main__":
    main()
