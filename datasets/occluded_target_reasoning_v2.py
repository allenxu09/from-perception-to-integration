"""Build canonical matched pairs for occluded red-circle comparison."""

from __future__ import annotations

import bisect
import hashlib
import itertools
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from vlm_core.io import file_fingerprint, write_json, write_jsonl


OUTPUT_DIR = Path("paper/data/occluded_target_reasoning_v2")
PAIR_COUNT = 500
SEED = 260808
VERSION = "2.0.0"
CANVAS = (768, 512)
PATCH_SIZE = 104
COLORS = {
    "red": (220, 45, 45),
    "blue": (50, 105, 220),
    "outline": (42, 42, 42),
    "divider": (190, 190, 190),
}
OCCLUDER_COLORS = (
    (62, 66, 70),
    (142, 132, 111),
    (92, 112, 96),
    (116, 104, 128),
    (184, 178, 158),
)
NON_CIRCLES = ("triangle", "rectangle", "star", "pentagon", "diamond", "cross")
OCCLUDERS = ("vertical", "horizontal", "rotated_rectangle", "ellipse", "polygon", "double_bar")
QUESTION = (
    "Question: Which side has more red circles after completing the occluded shapes?\n"
    "A. left\n"
    "B. right\n"
    "Answer with one option only."
)


def main() -> None:
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_dir = OUTPUT_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    orientations = ["left"] * (PAIR_COUNT // 2) + ["right"] * (PAIR_COUNT // 2)
    rng.shuffle(orientations)
    test_pairs = set(rng.sample(range(PAIR_COUNT), PAIR_COUNT // 5))

    samples: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    splits: list[dict[str, str]] = []
    for index in range(PAIR_COUNT):
        pair_samples, pair = build_pair(index, orientations[index], image_dir, rng)
        samples.extend(pair_samples)
        pairs.append(pair)
        split = "test" if index in test_pairs else "train"
        splits.extend(
            {"sample_id": sample["sample_id"], "pair_id": pair["pair_id"], "split": split}
            for sample in pair_samples
        )

    write_jsonl(OUTPUT_DIR / "samples.jsonl", samples)
    write_jsonl(OUTPUT_DIR / "pairs.jsonl", pairs)
    write_jsonl(OUTPUT_DIR / "splits.jsonl", splits)
    fingerprint = file_fingerprint((OUTPUT_DIR / "samples.jsonl", OUTPUT_DIR / "pairs.jsonl", OUTPUT_DIR / "splits.jsonl"))
    write_json(
        OUTPUT_DIR / "manifest.json",
        {
            "dataset_id": "occluded_target_reasoning",
            "version": VERSION,
            "seed": SEED,
            "generator": "datasets/occluded_target_reasoning_v2.py",
            "sample_count": len(samples),
            "pair_count": len(pairs),
            "pairing": "matched_color_shape_binding_counterfactual",
            "samples_path": "samples.jsonl",
            "pairs_path": "pairs.jsonl",
            "splits_path": "splits.jsonl",
            "split": {"train_pairs": 400, "test_pairs": 100, "grouped_by": "pair_id"},
            "config": {
                "canvas": list(CANVAS),
                "objects_per_side": [8, 12],
                "occlusion_ratio": [0.25, 0.60],
                "colors": ["red", "blue"],
                "shapes": ["circle", *NON_CIRCLES],
                "changed_factor": "red_circle_color_shape_binding",
                "shared_within_pair": ["positions", "shapes", "sizes", "rotations", "occluders", "marginal_counts"],
            },
            "fingerprint": fingerprint,
            "formal_experiments": [],
            "formal_results": [],
            "result_status": "dataset_only_pending_experiment_review",
        },
    )
    print(OUTPUT_DIR / "samples.jsonl")


def build_pair(
    index: int,
    a_larger_side: str,
    image_dir: Path,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_id = f"occluded_target_v2_{index:06d}"
    for _attempt in range(200):
        count = rng.randint(8, 12)
        circle_count = rng.randint(3, min(6, count - 3))
        red_count = rng.randint(3, min(6, count - 3))
        minimum = max(1, red_count - (count - circle_count))
        maximum = min(red_count, circle_count)
        if maximum - minimum < 1:
            continue
        low = rng.randint(minimum, maximum - 1)
        high = rng.randint(low + 1, maximum)
        slots = build_slots(count, circle_count, rng)
        try:
            high_red, low_red, red_area_gap = matched_red_sets(slots, red_count, high, low, 1 if index % 2 == 0 else -1)
        except ValueError:
            continue
        total_visible = sum(slot["visible_area"] for slot in slots)
        if abs(red_area_gap) <= max(18, round(total_visible * 0.006)):
            break
    else:
        raise RuntimeError(f"Could not build pixel-matched pair {pair_id}.")

    assignments = {
        "left": high_red if a_larger_side == "left" else low_red,
        "right": low_red if a_larger_side == "left" else high_red,
    }
    opposite = "right" if a_larger_side == "left" else "left"
    a = make_sample(pair_id, "a", a_larger_side, slots, assignments, image_dir)
    b = make_sample(
        pair_id,
        "b",
        opposite,
        slots,
        {"left": assignments["right"], "right": assignments["left"]},
        image_dir,
    )
    pair = {
        "pair_id": pair_id,
        "primitive": "multi_primitive_fusion_occlusion",
        "subtype": "occluded_target_reasoning_v2",
        "sample_a": a["sample_id"],
        "sample_b": b["sample_id"],
        "answer_a": a["answer"],
        "answer_b": b["answer"],
        "semantic_a": a_larger_side,
        "semantic_b": opposite,
        "changed_factor": "red_circle_color_shape_binding",
        "shared_scene_id": pair_id,
        "marginals": {
            "objects_per_side": count,
            "red_per_side": red_count,
            "circles_per_side": circle_count,
            "target_low": low,
            "target_high": high,
        },
        "red_visible_area_gap_high_minus_low": red_area_gap,
    }
    return [a, b], pair


def build_slots(count: int, circle_count: int, rng: random.Random) -> list[dict[str, Any]]:
    shapes = ["circle"] * circle_count
    while len(shapes) < count:
        shapes.append(NON_CIRCLES[(len(shapes) - circle_count) % len(NON_CIRCLES)])
    rng.shuffle(shapes)
    centers_left = random_centers(count, rng, 54, 330)
    centers_right = random_centers(count, rng, CANVAS[0] - 330, CANVAS[0] - 54)
    slots = []
    for index, shape in enumerate(shapes):
        for _attempt in range(100):
            size = rng.randint(29, 35)
            angle = rng.randrange(0, 360, 15)
            occluder_type = rng.choice(OCCLUDERS)
            occluder_scale = rng.uniform(0.34, 0.62)
            occluder_angle = rng.randrange(0, 180, 10)
            occluder_color = rng.choice(OCCLUDER_COLORS)
            shape_mask = make_shape_mask(shape, size, angle)
            occluder_mask = make_occluder_mask(
                occluder_type,
                size,
                occluder_scale,
                rng.randrange(-18, 19, 3),
                occluder_angle,
            )
            full_area = mask_area(shape_mask)
            covered = mask_intersection_area(shape_mask, occluder_mask)
            ratio = covered / full_area
            if 0.25 <= ratio <= 0.60:
                break
        else:
            raise RuntimeError("Could not sample valid occlusion.")
        slots.append(
            {
                "slot": index,
                "center_left": centers_left[index],
                "center_right": centers_right[index],
                "mirror_right": bool(rng.getrandbits(1)),
                "shape": shape,
                "size": size,
                "rotation": angle,
                "occluder_type": occluder_type,
                "occluder_scale": round(occluder_scale, 6),
                "occluder_offset": int(occluder_mask.info["offset"]),
                "occluder_angle": occluder_angle,
                "occluder_color": list(occluder_color),
                "occlusion_ratio": round(ratio, 6),
                "full_area": full_area,
                "visible_area": full_area - covered,
            }
        )
    return slots


def matched_red_sets(
    slots: list[dict[str, Any]],
    red_count: int,
    high: int,
    low: int,
    desired_sign: int,
) -> tuple[set[int], set[int], int]:
    circles = [slot["slot"] for slot in slots if slot["shape"] == "circle"]
    others = [slot["slot"] for slot in slots if slot["shape"] != "circle"]
    areas = {slot["slot"]: int(slot["visible_area"]) for slot in slots}

    def candidates(target: int) -> list[tuple[int, frozenset[int]]]:
        rows = []
        for chosen_circles in itertools.combinations(circles, target):
            for chosen_others in itertools.combinations(others, red_count - target):
                chosen = frozenset((*chosen_circles, *chosen_others))
                rows.append((sum(areas[index] for index in chosen), chosen))
        return sorted(rows, key=lambda row: row[0])

    high_rows = candidates(high)
    low_rows = candidates(low)
    low_areas = [row[0] for row in low_rows]
    best: tuple[int, frozenset[int], frozenset[int]] | None = None
    for high_area, high_set in high_rows:
        position = bisect.bisect_left(low_areas, high_area)
        for candidate in range(max(0, position - 3), min(len(low_rows), position + 4)):
            low_area, low_set = low_rows[candidate]
            gap = high_area - low_area
            if gap != 0 and int(math.copysign(1, gap)) != desired_sign:
                continue
            score = abs(gap)
            if best is None or score < best[0]:
                best = (score, high_set, low_set)
    if best is None:
        raise ValueError("No compatible red assignments.")
    high_set, low_set = set(best[1]), set(best[2])
    return high_set, low_set, sum(areas[index] for index in high_set) - sum(areas[index] for index in low_set)


def make_sample(
    pair_id: str,
    side: str,
    larger_side: str,
    slots: list[dict[str, Any]],
    red_slots: dict[str, set[int]],
    image_dir: Path,
) -> dict[str, Any]:
    image_path = image_dir / f"{pair_id}_{side}.png"
    objects, pixel_stats = render_scene(image_path, slots, red_slots)
    counts = scene_counts(objects)
    answer = "A" if larger_side == "left" else "B"
    return {
        "sample_id": f"{pair_id}_{side}",
        "pair_id": pair_id,
        "counterfactual_id": pair_id,
        "primitive": "multi_primitive_fusion_occlusion",
        "subtask": "occluded_target_reasoning_v2",
        "difficulty": "controlled",
        "image_path": f"images/{image_path.name}",
        "question": QUESTION,
        "choices": ["A", "B"],
        "answer": answer,
        "source": "canonical_occluded_target_reasoning_v2",
        "metadata": {
            **counts,
            "larger_target_side": larger_side,
            "pair_side": side,
            "changed_factor": "red_circle_color_shape_binding",
            "shared_scene_id": pair_id,
            "pixel_stats": pixel_stats,
            "objects": objects,
        },
    }


def render_scene(
    path: Path,
    slots: list[dict[str, Any]],
    red_slots: dict[str, set[int]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    image = Image.new("RGB", CANVAS, "white")
    draw = ImageDraw.Draw(image)
    draw.line((CANVAS[0] // 2, 32, CANVAS[0] // 2, CANVAS[1] - 32), fill=COLORS["divider"], width=4)
    objects = []
    for side in ("left", "right"):
        for slot in slots:
            right_side = side == "right"
            mirror = right_side and bool(slot["mirror_right"])
            shape_mask = make_shape_mask(slot["shape"], slot["size"], slot["rotation"])
            occluder_mask = make_occluder_mask(
                slot["occluder_type"],
                slot["size"],
                float(slot["occluder_scale"]),
                int(slot["occluder_offset"]),
                int(slot["occluder_angle"]),
            )
            if mirror:
                shape_mask = ImageOps.mirror(shape_mask)
                occluder_mask = ImageOps.mirror(occluder_mask)
            center = tuple(slot["center_right"] if right_side else slot["center_left"])
            color = "red" if slot["slot"] in red_slots[side] else "blue"
            paste_object(image, center, shape_mask, occluder_mask, COLORS[color], tuple(slot["occluder_color"]))
            objects.append(
                {
                    "side": side,
                    "slot": slot["slot"],
                    "center": list(center),
                    "shape": slot["shape"],
                    "color": color,
                    "size": slot["size"],
                    "rotation": (-slot["rotation"]) % 360 if mirror else slot["rotation"],
                    "occluder_type": slot["occluder_type"],
                    "occluder_angle": (-slot["occluder_angle"]) % 180 if mirror else slot["occluder_angle"],
                    "occluder_color": slot["occluder_color"],
                    "occlusion_ratio": slot["occlusion_ratio"],
                    "full_area": slot["full_area"],
                    "visible_area": slot["visible_area"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return objects, image_pixel_stats(image)


def paste_object(
    image: Image.Image,
    center: tuple[int, int],
    shape_mask: Image.Image,
    occluder_mask: Image.Image,
    color: tuple[int, int, int],
    occluder_color: tuple[int, int, int],
) -> None:
    x = center[0] - PATCH_SIZE // 2
    y = center[1] - PATCH_SIZE // 2
    outline = shape_mask.filter(ImageFilter.MaxFilter(5))
    image.paste(COLORS["outline"], (x, y), outline)
    image.paste(color, (x, y), shape_mask)
    image.paste(occluder_color, (x, y), occluder_mask)


def make_shape_mask(shape: str, size: int, angle: int) -> Image.Image:
    mask = Image.new("L", (PATCH_SIZE, PATCH_SIZE))
    draw = ImageDraw.Draw(mask)
    cx = cy = PATCH_SIZE // 2
    if shape == "circle":
        draw.ellipse((cx - size, cy - size, cx + size, cy + size), fill=255)
    elif shape == "rectangle":
        draw.polygon(rotated_regular_polygon(cx, cy, size, 4, angle + 45), fill=255)
    elif shape == "triangle":
        draw.polygon(rotated_regular_polygon(cx, cy, size * 1.12, 3, angle - 90), fill=255)
    elif shape == "star":
        draw.polygon(star_points(cx, cy, size * 1.12, angle), fill=255)
    elif shape == "pentagon":
        draw.polygon(rotated_regular_polygon(cx, cy, size * 1.05, 5, angle - 90), fill=255)
    elif shape == "diamond":
        draw.polygon(rotated_regular_polygon(cx, cy, size * 1.08, 4, angle), fill=255)
    elif shape == "cross":
        draw.polygon(cross_points(cx, cy, size, angle), fill=255)
    else:
        raise ValueError(shape)
    return mask


def make_occluder_mask(kind: str, size: int, scale: float, offset: int, angle: int) -> Image.Image:
    mask = Image.new("L", (PATCH_SIZE, PATCH_SIZE))
    draw = ImageDraw.Draw(mask)
    cx = cy = PATCH_SIZE // 2
    width = max(18, round(2 * size * scale))
    if kind == "vertical":
        draw.rectangle((cx + offset - width // 2, cy - size - 5, cx + offset + width // 2, cy + size + 5), fill=255)
    elif kind == "horizontal":
        draw.rectangle((cx - size - 5, cy + offset - width // 2, cx + size + 5, cy + offset + width // 2), fill=255)
    elif kind == "rotated_rectangle":
        draw.polygon(rotated_rectangle(cx, cy, 2.7 * size, width, angle), fill=255)
    elif kind == "ellipse":
        radius = max(14, round(size * scale))
        draw.ellipse((cx + offset - radius, cy - radius, cx + offset + radius, cy + radius), fill=255)
    elif kind == "polygon":
        radius = max(16, round(size * scale * 1.35))
        draw.polygon(rotated_regular_polygon(cx + offset, cy, radius, 6, angle), fill=255)
    elif kind == "double_bar":
        bar_width = max(10, width // 2)
        draw.polygon(rotated_rectangle(cx + offset, cy - size * 0.22, 2.5 * size, bar_width, angle), fill=255)
        draw.polygon(rotated_rectangle(cx - offset, cy + size * 0.22, 2.5 * size, bar_width, angle), fill=255)
    else:
        raise ValueError(kind)
    mask.info["offset"] = offset
    return mask


def random_centers(count: int, rng: random.Random, x_min: int, x_max: int) -> list[tuple[int, int]]:
    for _restart in range(100):
        centers: list[tuple[int, int]] = []
        for _ in range(3000):
            candidate = (rng.randint(x_min, x_max), rng.randint(54, CANVAS[1] - 54))
            if all((candidate[0] - x) ** 2 + (candidate[1] - y) ** 2 >= 80**2 for x, y in centers):
                centers.append(candidate)
                if len(centers) == count:
                    return centers
    raise RuntimeError("Could not sample non-overlapping object centers.")


def rotated_regular_polygon(cx: float, cy: float, radius: float, sides: int, angle: float) -> list[tuple[int, int]]:
    start = math.radians(angle)
    return [
        (round(cx + radius * math.cos(start + 2 * math.pi * index / sides)), round(cy + radius * math.sin(start + 2 * math.pi * index / sides)))
        for index in range(sides)
    ]


def star_points(cx: float, cy: float, radius: float, angle: float) -> list[tuple[int, int]]:
    start = math.radians(angle - 90)
    return [
        (
            round(cx + (radius if index % 2 == 0 else radius * 0.45) * math.cos(start + index * math.pi / 5)),
            round(cy + (radius if index % 2 == 0 else radius * 0.45) * math.sin(start + index * math.pi / 5)),
        )
        for index in range(10)
    ]


def cross_points(cx: float, cy: float, radius: float, angle: float) -> list[tuple[int, int]]:
    arm = radius * 0.38
    points = [
        (-arm, -radius), (arm, -radius), (arm, -arm), (radius, -arm),
        (radius, arm), (arm, arm), (arm, radius), (-arm, radius),
        (-arm, arm), (-radius, arm), (-radius, -arm), (-arm, -arm),
    ]
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    return [
        (round(cx + x * cosine - y * sine), round(cy + x * sine + y * cosine))
        for x, y in points
    ]


def rotated_rectangle(cx: float, cy: float, length: float, width: float, angle: float) -> list[tuple[int, int]]:
    radians = math.radians(angle)
    ux, uy = math.cos(radians), math.sin(radians)
    vx, vy = -uy, ux
    return [
        (round(cx + ux * x + vx * y), round(cy + uy * x + vy * y))
        for x, y in ((-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2), (-length / 2, width / 2))
    ]


def mask_area(mask: Image.Image) -> int:
    return mask.width * mask.height - mask.histogram()[0]


def mask_intersection_area(left: Image.Image, right: Image.Image) -> int:
    intersection = ImageChops.multiply(left, right)
    return intersection.width * intersection.height - intersection.histogram()[0]


def scene_counts(objects: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for side in ("left", "right"):
        rows = [obj for obj in objects if obj["side"] == side]
        counts[f"{side}_total_count"] = len(rows)
        counts[f"{side}_red_count"] = sum(obj["color"] == "red" for obj in rows)
        counts[f"{side}_blue_count"] = sum(obj["color"] == "blue" for obj in rows)
        for shape in ("circle", *NON_CIRCLES):
            counts[f"{side}_{shape}_count"] = sum(obj["shape"] == shape for obj in rows)
        counts[f"{side}_target_count"] = sum(obj["color"] == "red" and obj["shape"] == "circle" for obj in rows)
        counts[f"{side}_mean_occlusion"] = round(sum(float(obj["occlusion_ratio"]) for obj in rows) / len(rows), 6)
        target = [obj for obj in rows if obj["color"] == "red" and obj["shape"] == "circle"]
        counts[f"{side}_target_mean_occlusion"] = round(sum(float(obj["occlusion_ratio"]) for obj in target) / len(target), 6)
    return counts


def image_pixel_stats(image: Image.Image) -> dict[str, int]:
    stats: dict[str, int] = {}
    boxes = {"left": (0, 0, CANVAS[0] // 2, CANVAS[1]), "right": (CANVAS[0] // 2 + 1, 0, CANVAS[0], CANVAS[1])}
    for side, box in boxes.items():
        crop = image.crop(box)
        counts = dict(crop.getcolors(crop.width * crop.height) or [])
        stats[f"{side}_red_pixels"] = counts.get(COLORS["red"], 0)
        stats[f"{side}_blue_pixels"] = counts.get(COLORS["blue"], 0)
        stats[f"{side}_foreground_pixels"] = crop.width * crop.height - counts.get((255, 255, 255), 0)
        stats[f"{side}_occluder_pixels"] = sum(counts.get(color, 0) for color in OCCLUDER_COLORS)
    return stats


if __name__ == "__main__":
    main()
