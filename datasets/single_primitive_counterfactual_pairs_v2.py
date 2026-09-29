"""Build the canonical four-primitive matched counterfactual pairs."""

from __future__ import annotations

import math
import random
import sys
from collections import Counter, deque
from heapq import heapify, heappop, heappush
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_core.io import file_fingerprint, write_json, write_jsonl

from single_primitive_counterfactual_pairs import COLORS, SHAPES, draw_shape


DATASET_ID = "single_primitive_counterfactual_pairs"
VERSION = "2.1.0"
SEED = 4041
PAIRS_PER_PRIMITIVE = 125
OUTPUT_DIR = ROOT / "paper" / "data" / DATASET_ID
PREVIEW_PAIR_COUNT = 30
PREVIEW_DIR = ROOT / "paper" / "previews" / "amodal_completion_controlled_pairs_30"
COLOR_OPTIONS = tuple(COLORS)
COLOR_ANSWER = {value: chr(ord("A") + index) for index, value in enumerate(COLOR_OPTIONS)}
RELATIONS = ("left", "right", "above", "below")
RELATION_ANSWER = {value: chr(ord("A") + index) for index, value in enumerate(RELATIONS)}
ATTRIBUTE_PAPER_NAME = "Targeted Attribute Binding"


def main() -> None:
    manifest = build_dataset(OUTPUT_DIR)
    print(OUTPUT_DIR / "manifest.json")
    print(manifest["fingerprint"])


def build_dataset(output_dir: Path) -> dict[str, Any]:
    rng = random.Random(SEED)
    for path in output_dir.glob("*/images/*.png"):
        path.unlink()
    schedules = {
        "attribute": balanced_pairs(COLOR_OPTIONS, [63, 63, 62, 62], rng),
        "spatial": balanced_pairs(RELATIONS, [63, 63, 62, 62], rng),
        "numerosity": [("left", "right")] * PAIRS_PER_PRIMITIVE,
        "occlusion": continuation_pairs(PAIRS_PER_PRIMITIVE, rng),
    }
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    builders = {
        "attribute": build_attribute_pair,
        "numerosity": build_numerosity_pair,
        "spatial": build_spatial_pair,
        "occlusion": build_occlusion_pair,
    }
    for primitive, builder in builders.items():
        labels = schedules[primitive] if primitive == "occlusion" else orient_pairs(schedules[primitive], rng)
        if primitive == "occlusion":
            rng.shuffle(labels)
        for index, endpoints in enumerate(labels):
            members, pair = builder(index, endpoints, output_dir / primitive / "images", rng)
            rows.extend(members)
            pairs.append(pair)
    test_pairs = set()
    for primitive in builders:
        groups: dict[str, list[str]] = {}
        for pair in pairs:
            if pair["primitive_key"] == primitive:
                groups.setdefault(pair["clean_answer"], []).append(pair["pair_id"])
        for index, pair_ids in enumerate(sorted(groups.values())):
            rng.shuffle(pair_ids)
            test_pairs.update(pair_ids[: 25 // len(groups) + (1 if index < 25 % len(groups) else 0)])
    splits = [
        {"sample_id": row["sample_id"], "pair_id": row["pair_id"], "split": "test" if row["pair_id"] in test_pairs else "train"}
        for row in rows
    ]
    validate(rows, pairs, splits)
    config = {
        "dataset_id": DATASET_ID,
        "version": VERSION,
        "seed": SEED,
        "sample_count": len(rows),
        "pair_count": len(pairs),
        "pairs_per_primitive": PAIRS_PER_PRIMITIVE,
        "pairing": "matched_counterfactual",
        "split": {"train_pairs_per_primitive": 100, "test_pairs_per_primitive": 25, "grouped_by": "pair_id"},
    }
    write_json(output_dir / "config.json", config)
    write_jsonl(output_dir / "samples.jsonl", rows)
    write_jsonl(output_dir / "pairs.jsonl", pairs)
    write_jsonl(output_dir / "splits.jsonl", splits)
    fingerprint = file_fingerprint(
        [
            output_dir / "config.json",
            output_dir / "samples.jsonl",
            output_dir / "pairs.jsonl",
            output_dir / "splits.jsonl",
            *(ROOT / row["image_path"] for row in rows),
        ]
    )
    manifest = {
        **config,
        "generator": "datasets/single_primitive_counterfactual_pairs_v2.py",
        "helper": "datasets/single_primitive_counterfactual_pairs.py",
        "samples_path": "samples.jsonl",
        "pairs_path": "pairs.jsonl",
        "splits_path": "splits.jsonl",
        "fingerprint": fingerprint,
        "formal_experiments": [
            "experiments/counterfactual_activation_patching.py",
            "experiments/counterfactual_gradient_attribution.py",
            "experiments/pathway_availability.py",
            "experiments/rescue_set_decodability.py",
            "experiments/primitive_jspace_intervention.py",
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


def orient_pairs(pairs: list[tuple[str, str]], rng: random.Random) -> list[tuple[str, str]]:
    pairs = pairs[:]
    rng.shuffle(pairs)
    clean_counts: Counter[str] = Counter()
    output = []
    for left, right in pairs:
        if clean_counts[left] > clean_counts[right] or (clean_counts[left] == clean_counts[right] and rng.random() < 0.5):
            left, right = right, left
        clean_counts[left] += 1
        output.append((left, right))
    return output


def continuation_pairs(count: int, rng: random.Random) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    family_values = ("circle", "ellipse", "rounded_rectangle", "triangle", "polygon", "arc", "s_curve", "smooth_contour")
    families = [family_values[index % len(family_values)] for index in range(count)]
    colors = balanced_values(COLOR_OPTIONS, count, rng)
    thicknesses = balanced_values((6, 8, 10, 12), count, rng)
    occluders = balanced_values(("rectangle", "rotated_rectangle", "ellipse", "polygon"), count, rng)
    specs = [make_underlying_spec(index, families[index], colors[index], thicknesses[index], occluders[index], rng) for index in range(count)]
    donors = [-1] * count
    for closed in (True, False):
        group = [index for index, spec in enumerate(specs) if spec["closed"] == closed]
        shifted = group[1:] + group[:1]
        for index, donor in zip(group, shifted):
            donors[index] = donor
    return [(specs[index], specs[donor]) for index, donor in enumerate(donors)]


def balanced_values(values: tuple[Any, ...], count: int, rng: random.Random) -> list[Any]:
    output = [values[index % len(values)] for index in range(count)]
    rng.shuffle(output)
    return output


def make_underlying_spec(index: int, family: str, color: str, thickness: int, occluder_type: str, rng: random.Random) -> dict[str, Any]:
    closed = family not in {"arc", "s_curve"}
    return {
        "geometry_id": f"underlying_geometry_{index:04d}", "family": family, "closed": closed,
        "color": color, "thickness": thickness, "occluder_type": occluder_type,
        "center": (rng.randint(210, 238), rng.randint(166, 194)),
        "width": rng.randint(235, 285), "height": rng.randint(155, 205),
        "rotation": rng.uniform(-24, 24), "phase": rng.uniform(-0.35, 0.35),
    }


def normalized_geometry(spec: dict[str, Any]) -> list[tuple[float, float]]:
    family = spec["family"]
    count = 721 if spec["closed"] else 241
    points = []
    if family in {"circle", "ellipse", "smooth_contour"}:
        for index in range(count):
            angle = 2 * math.pi * index / (count - 1)
            radius = 1.0 if family != "smooth_contour" else 1 + 0.10 * math.sin(3 * angle + spec["phase"])
            y_scale = 1.0 if family == "circle" else 0.82
            points.append((radius * math.cos(angle), y_scale * radius * math.sin(angle)))
    elif family == "rounded_rectangle":
        for index in range(count):
            angle = 2 * math.pi * index / (count - 1)
            cosine, sine = math.cos(angle), math.sin(angle)
            points.append((math.copysign(abs(cosine) ** 0.65, cosine), 0.82 * math.copysign(abs(sine) ** 0.65, sine)))
    elif family in {"triangle", "polygon"}:
        sides = 3 if family == "triangle" else 5
        vertices = [(math.cos(-math.pi / 2 + 2 * math.pi * i / sides), 0.86 * math.sin(-math.pi / 2 + 2 * math.pi * i / sides)) for i in range(sides)]
        segment_count = (count - 1) // sides
        for side in range(sides):
            start, end = vertices[side], vertices[(side + 1) % sides]
            for step in range(segment_count):
                t = step / segment_count
                points.append((start[0] * (1 - t) + end[0] * t, start[1] * (1 - t) + end[1] * t))
        points.append(points[0])
    elif family == "arc":
        for index in range(count):
            t = -1 + 2 * index / (count - 1)
            points.append((t, 0.72 * t * t - 0.34))
    else:
        for index in range(count):
            t = -1 + 2 * index / (count - 1)
            points.append((t, 0.55 * math.sin(math.pi * t)))
    return points


def transform_points(points: list[tuple[float, float]], spec: dict[str, Any]) -> list[tuple[float, float]]:
    angle = math.radians(spec["rotation"])
    cosine, sine = math.cos(angle), math.sin(angle)
    cx, cy = spec["center"]
    return [
        (cx + x * spec["width"] / 2 * cosine - y * spec["height"] / 2 * sine,
         cy + x * spec["width"] / 2 * sine + y * spec["height"] / 2 * cosine)
        for x, y in points
    ]


def corrupt_geometry(clean_spec: dict[str, Any], donor_spec: dict[str, Any]) -> list[tuple[float, float]]:
    clean = normalized_geometry(clean_spec)
    donor = normalized_geometry(donor_spec)
    output = []
    for (x, y), (donor_x, donor_y) in zip(clean, donor):
        weight = max(0.0, min(1.0, (x + 0.32) / 0.64))
        weight = weight * weight * (3 - 2 * weight)
        output.append((x * (1 - weight) + donor_x * weight, y * (1 - weight) + donor_y * weight))
    return output


def continuation_occluder(spec: dict[str, Any], rng: random.Random) -> tuple[Image.Image, dict[str, Any]]:
    cx, cy = spec["center"]
    kind = spec["occluder_type"]
    width = rng.randint(82, 112)
    height = int(spec["height"] * rng.uniform(1.10, 1.28))
    angle = spec["rotation"] + rng.uniform(-12, 12)
    mask = Image.new("L", (448, 360), 0)
    draw = ImageDraw.Draw(mask)
    if kind == "rectangle":
        draw.rectangle((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), fill=255)
        angle = 0.0
    elif kind == "ellipse":
        draw.ellipse((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), fill=255)
        angle = 0.0
    elif kind == "rotated_rectangle":
        draw.polygon(rotated_box((cx, cy), width, height, angle), fill=255)
    else:
        points = rotated_box((cx, cy), width, height, angle)
        points[0] = (points[0][0] + rng.uniform(-12, 12), points[0][1] + rng.uniform(-12, 12))
        points[2] = (points[2][0] + rng.uniform(-12, 12), points[2][1] + rng.uniform(-12, 12))
        draw.polygon(points, fill=255)
    return mask, {"center": [cx, cy], "width": width, "height": height, "angle_degrees": round(angle, 3)}


def rotated_box(center: tuple[float, float], width: float, height: float, angle: float) -> list[tuple[float, float]]:
    cx, cy = center
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return [
        (cx + x * cosine - y * sine, cy + x * sine + y * cosine)
        for x, y in ((-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2))
    ]


def target_mask(points: list[tuple[float, float]], thickness: int, closed: bool) -> Image.Image:
    mask = Image.new("L", (448, 360), 0)
    draw = ImageDraw.Draw(mask)
    draw.line(points, fill=255, width=thickness, joint="curve")
    if closed:
        draw.line((points[-1], points[0]), fill=255, width=thickness, joint="curve")
    return mask


def component_sizes(mask: Image.Image) -> list[int]:
    pixels = mask.load()
    seen = set()
    sizes = []
    box = mask.getbbox()
    if box is None:
        return sizes
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            if not pixels[x, y] or (x, y) in seen:
                continue
            queue = deque([(x, y)])
            seen.add((x, y))
            size = 0
            while queue:
                px, py = queue.popleft()
                size += 1
                for neighbor in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                    nx, ny = neighbor
                    if box[0] <= nx < box[2] and box[1] <= ny < box[3] and pixels[nx, ny] and neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            sizes.append(size)
    return sorted(sizes, reverse=True)


def build_attribute_pair(index: int, labels: tuple[str, str], image_dir: Path, rng: random.Random):
    pair_id = f"attribute_pair_v2_{index:04d}"
    positions = rng.sample([(x, y) for y in (70, 180, 290) for x in (62, 170, 278, 386)], 8)
    size = rng.choice((34, 40, 46))
    shapes = [shape for shape in SHAPES for _ in range(2)]
    rng.shuffle(shapes)
    swap_index = shapes.index(shapes[0], 1)
    remaining_colors = [color for color in COLOR_OPTIONS for _ in range(2)]
    remaining_colors.remove(labels[0])
    remaining_colors.remove(labels[1])
    rng.shuffle(remaining_colors)
    remaining = iter(remaining_colors)
    colors = [labels[0] if object_index == 0 else labels[1] if object_index == swap_index else next(remaining) for object_index in range(8)]
    objects = [
        {"role": "distractor", "center": center, "shape": shape, "color": color, "outlined": False}
        for center, shape, color in zip(positions, shapes, colors)
    ]
    objects[0].update(role="target", outlined=True, outline="black")
    corrupt_objects = [dict(obj) for obj in objects]
    corrupt_objects[0]["color"], corrupt_objects[swap_index]["color"] = (
        corrupt_objects[swap_index]["color"], corrupt_objects[0]["color"]
    )
    scenes = {"clean": objects, "corrupt": corrupt_objects}
    question = "Question: What color is the outlined object?\nA. red\nB. blue\nC. green\nD. yellow\nAnswer with one option only."
    members = []
    for role, color in zip(("clean", "corrupt"), labels):
        scene = scenes[role]
        path = image_dir / f"{pair_id}_{role}.png"
        bboxes = draw_objects(path, scene, size)
        metadata = {
            "paper_name": ATTRIBUTE_PAPER_NAME,
            "target_color": color,
            "target_shape": shapes[0],
            "target_bbox": list(bboxes[0]),
            "object_size": size,
            "color_swap_indices": [0, swap_index],
            "objects": [{**obj, "center": list(obj["center"]), "bbox": list(box)} for obj, box in zip(scene, bboxes)],
        }
        members.append(sample_row(pair_id, role, "attribute", "attribute_outlined_target_cf_v2", path, question, ["A", "B", "C", "D"], COLOR_ANSWER[color], "outlined_target_color", metadata))
    return members, pair_row(pair_id, "attribute", members, "outlined_target_color")


def build_numerosity_pair(index: int, labels: tuple[str, str], image_dir: Path, rng: random.Random):
    pair_id = f"numerosity_pair_v2_{index:04d}"
    ratio = rng.choice((1.25, 1.5, 2.0))
    smaller = rng.choice((8, 10, 12, 14))
    larger = round(smaller * ratio)
    left_layout = dot_layout(larger, (128, 128), rng)
    right_layout = dot_layout(smaller, (384, 128), rng)
    layouts = {
        "left": (left_layout, right_layout),
        "right": ([(x - 256, y) for x, y in right_layout], [(x + 256, y) for x, y in left_layout]),
    }
    question = "Question: Which side has more dots?\nA. left\nB. right\nAnswer with one option only."
    members = []
    for role, larger_side in zip(("clean", "corrupt"), labels):
        left, right = layouts[larger_side]
        path = image_dir / f"{pair_id}_{role}.png"
        draw_dots(path, left, right)
        metadata = {
            "left_count": len(left), "right_count": len(right), "larger_count": larger,
            "larger_side": larger_side, "ratio": ratio,
            "left_points": [list(point) for point in left], "right_points": [list(point) for point in right],
            "dot_radius": 7,
        }
        members.append(sample_row(pair_id, role, "numerosity_individuation", "nasco_dot_array_cf_v2", path, question, ["A", "B"], "A" if larger_side == "left" else "B", "larger_side", metadata))
    return members, pair_row(pair_id, "numerosity", members, "larger_side")


def build_spatial_pair(index: int, labels: tuple[str, str], image_dir: Path, rng: random.Random):
    pair_id = f"spatial_pair_v2_{index:04d}"
    distance = rng.choice((70, 90, 110))
    size = rng.choice((24, 28, 32))
    margin = distance + size + 7
    reference = (rng.randint(margin, 448 - margin), rng.randint(margin, 360 - margin))
    subject_centers = {relation: relation_center(reference, relation, distance) for relation in labels}
    occupied = [reference, *subject_centers.values()]
    distractor_centers = []
    while len(distractor_centers) < 4:
        point = (rng.randint(50, 398), rng.randint(50, 310))
        if all(math.dist(point, other) >= 2 * size + 24 for other in [*occupied, *distractor_centers]):
            distractor_centers.append(point)
    shared = {
        "subject_shape": rng.choice(SHAPES), "subject_color": rng.choice(COLOR_OPTIONS),
        "reference_shape": rng.choice(SHAPES), "reference_color": rng.choice(COLOR_OPTIONS),
    }
    distractors = [
        {"role": "distractor", "center": center, "shape": rng.choice(SHAPES), "color": rng.choice(COLOR_OPTIONS), "outlined": False}
        for center in distractor_centers
    ]
    question = "Question: Where is the black-marked object relative to the gray-marked object?\nA. left\nB. right\nC. above\nD. below\nAnswer with one option only."
    members = []
    for role, relation in zip(("clean", "corrupt"), labels):
        objects = [
            {"role": "subject", "center": subject_centers[relation], "shape": shared["subject_shape"], "color": shared["subject_color"], "outlined": True, "outline": "black"},
            {"role": "reference", "center": reference, "shape": shared["reference_shape"], "color": shared["reference_color"], "outlined": True, "outline": "gray"},
            *distractors,
        ]
        path = image_dir / f"{pair_id}_{role}.png"
        bboxes = draw_objects(path, objects, size)
        metadata = {
            "true_relation_label": relation, "distance": distance, "object_size": size,
            "subject_bbox": list(bboxes[0]), "reference_bbox": list(bboxes[1]),
            "objects": [{**obj, "center": list(obj["center"]), "bbox": list(box)} for obj, box in zip(objects, bboxes)],
        }
        members.append(sample_row(pair_id, role, "spatial_relation", "spatial_marked_relation_cf_v2", path, question, ["A", "B", "C", "D"], RELATION_ANSWER[relation], "marked_object_relation", metadata))
    return members, pair_row(pair_id, "spatial", members, "marked_object_relation")


def build_occlusion_pair(index: int, specs: tuple[dict[str, Any], dict[str, Any]], image_dir: Path, rng: random.Random):
    pair_id = f"occlusion_pair_v2_{index:04d}"
    clean_spec, donor_spec = specs
    clean_normalized = normalized_geometry(clean_spec)
    corrupt_normalized = corrupt_geometry(clean_spec, donor_spec)
    clean_points = transform_points(clean_normalized, clean_spec)
    corrupt_points = transform_points(corrupt_normalized, clean_spec)
    occluder_mask, occluder_params = continuation_occluder(clean_spec, rng)
    clean_mask = target_mask(clean_points, clean_spec["thickness"], clean_spec["closed"])
    corrupt_mask = target_mask(corrupt_points, clean_spec["thickness"], clean_spec["closed"])
    clean_visible = ImageChops.subtract(clean_mask, occluder_mask)
    corrupt_visible = ImageChops.subtract(corrupt_mask, occluder_mask)
    clean_components = component_sizes(clean_visible)
    corrupt_components = component_sizes(corrupt_visible)
    if len(clean_components) < 2 or len(corrupt_components) < 2 or clean_components[1] < 100 or corrupt_components[1] < 100:
        raise RuntimeError(f"Occluder did not leave two substantial visible components for {pair_id}.")
    for points in (clean_points, corrupt_points):
        if max(math.dist(first, second) for first, second in zip(points, points[1:])) > 18:
            raise RuntimeError(f"Underlying geometry contains an abrupt jump for {pair_id}.")
        if any(not (8 <= x < 440 and 8 <= y < 352) for x, y in points):
            raise RuntimeError(f"Underlying geometry is clipped for {pair_id}.")
    question = "Question: Do the visible parts form a natural continuation behind the occluder?\nA. natural continuation\nB. incompatible continuation\nAnswer with one option only."
    members = []
    tags = ["a", "b"]
    rng.shuffle(tags)
    endpoints = (("clean", "natural", clean_points), ("corrupt", "incompatible", corrupt_points))
    for tag, (role, label_value, points) in zip(tags, endpoints):
        path = image_dir / f"{pair_id}_{role}.png"
        draw_continuation(path, points, clean_spec["closed"], occluder_mask, COLORS[clean_spec["color"]], clean_spec["thickness"])
        metadata = {
            "continuation_label": label_value,
            "visible_component_count": len(clean_components if role == "clean" else corrupt_components),
            "underlying_geometry_family": clean_spec["family"] if role == "clean" else f"{clean_spec['family']}+{donor_spec['family']}",
            "clean_geometry_id": clean_spec["geometry_id"], "donor_geometry_id": donor_spec["geometry_id"],
            "generation_order": "complete_underlying_geometry_then_occluder",
            "fragment_color": clean_spec["color"], "fragment_thickness": clean_spec["thickness"],
            "target_center": list(clean_spec["center"]), "target_rotation_degrees": clean_spec["rotation"],
            "target_width": clean_spec["width"], "target_height": clean_spec["height"],
            "occluder_type": clean_spec["occluder_type"], "occluder_params": occluder_params,
        }
        members.append(sample_row(pair_id, role, "occlusion_object_unity", "amodal_completion_preview_v3", path, question, ["A", "B"], "A" if label_value == "natural" else "B", "underlying_geometry", metadata, tag))
    return members, pair_row(pair_id, "occlusion", members, "underlying_geometry")


def sample_row(pair_id: str, role: str, primitive: str, subtask: str, image_path: Path, question: str, choices: list[str], answer: str, changed_factor: str, metadata: dict[str, Any], sample_suffix: str | None = None) -> dict[str, Any]:
    return {
        "sample_id": f"{pair_id}_{sample_suffix or role}", "pair_id": pair_id, "counterfactual_id": pair_id,
        "primitive": primitive, "subtask": subtask, "difficulty": "formal",
        "image_path": image_path.relative_to(ROOT).as_posix(), "question": question,
        "choices": choices, "answer": answer, "source": "single_primitive_counterfactual_pairs_v2",
        "metadata": {**metadata, "dataset_version": VERSION, "pair_id": pair_id, "clean_or_corrupt": role, "changed_factor": changed_factor},
    }


def pair_row(pair_id: str, primitive_key: str, members: list[dict[str, Any]], changed_factor: str) -> dict[str, Any]:
    clean, corrupt = members
    return {
        "pair_id": pair_id, "primitive_key": primitive_key, "primitive": clean["primitive"],
        "subtask": clean["subtask"], "changed_factor": changed_factor,
        "clean_sample_id": clean["sample_id"], "corrupt_sample_id": corrupt["sample_id"],
        "clean_answer": clean["answer"], "corrupt_answer": corrupt["answer"],
    }


def draw_objects(path: Path, objects: list[dict[str, Any]], size: int) -> list[tuple[int, int, int, int]]:
    image = Image.new("RGB", (448, 360), "white")
    draw = ImageDraw.Draw(image)
    boxes = []
    for obj in objects:
        cx, cy = obj["center"]
        box = (cx - size, cy - size, cx + size, cy + size)
        draw_shape(draw, obj["shape"], box, COLORS[obj["color"]])
        if obj.get("outlined"):
            outline = (0, 0, 0) if obj.get("outline", "black") == "black" else (130, 130, 130)
            draw.rectangle((box[0] - 6, box[1] - 6, box[2] + 6, box[3] + 6), outline=outline, width=5)
        boxes.append(box)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return boxes


def dot_layout(count: int, center: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    points = []
    while len(points) < count:
        point = (center[0] + rng.randint(-88, 88), center[1] + rng.randint(-88, 88))
        if all(math.dist(point, other) > 20 for other in points):
            points.append(point)
    return points


def draw_dots(path: Path, left: list[tuple[int, int]], right: list[tuple[int, int]]) -> None:
    image = Image.new("RGB", (512, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.line((256, 20, 256, 236), fill=(180, 180, 180), width=2)
    for x, y in left + right:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(35, 35, 35))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def relation_center(reference: tuple[int, int], relation: str, distance: int) -> tuple[int, int]:
    x, y = reference
    return {"left": (x - distance, y), "right": (x + distance, y), "above": (x, y - distance), "below": (x, y + distance)}[relation]


def draw_continuation(path: Path, points: list[tuple[float, float]], closed: bool, occluder: Image.Image, color: tuple[int, int, int], thickness: int) -> None:
    image = Image.new("RGB", (448, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.line(points, fill=color, width=thickness, joint="curve")
    if closed:
        draw.line((points[-1], points[0]), fill=color, width=thickness, joint="curve")
    image.paste((205, 205, 205), mask=occluder)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def build_amodal_preview(output_dir: Path) -> None:
    rng = random.Random(SEED)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for path in image_dir.glob("*.png"):
        path.unlink()
    rows = []
    pairs = []
    for index, specs in enumerate(continuation_pairs(PREVIEW_PAIR_COUNT, rng)):
        members, pair = build_occlusion_pair(index, specs, image_dir, rng)
        rows.extend(members)
        pairs.append(pair)
    write_jsonl(output_dir / "samples.jsonl", rows)
    write_jsonl(output_dir / "pairs.jsonl", pairs)
    write_json(output_dir / "config.json", {
        "status": "visual_review_only", "seed": SEED, "pair_count": PREVIEW_PAIR_COUNT,
        "generator": "datasets/single_primitive_counterfactual_pairs_v2.py",
        "generation_order": "complete_underlying_geometry_then_occluder",
    })
    make_contact_sheet(rows, output_dir / "contact_sheet.png")


def make_contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    by_pair = {rows[index]["pair_id"]: rows[index:index + 2] for index in range(0, len(rows), 2)}
    pair_width, pair_height, columns = 448, 205, 4
    sheet = Image.new("RGB", (pair_width * columns, pair_height * math.ceil(len(by_pair) / columns)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (pair_id, members) in enumerate(by_pair.items()):
        x, y = (index % columns) * pair_width, (index // columns) * pair_height
        draw.text((x + 4, y + 3), f"{pair_id}: clean | corrupt", fill="black")
        for side, member in enumerate(members):
            with Image.open(ROOT / member["image_path"]) as image:
                sheet.paste(image.resize((224, 180)), (x + 224 * side, y + 25))
    sheet.save(output_path)


def validate(rows: list[dict[str, Any]], pairs: list[dict[str, Any]], splits: list[dict[str, str]]) -> None:
    assert len(rows) == 1000 and len(pairs) == 500
    assert Counter(row["split"] for row in splits) == Counter({"train": 800, "test": 200})
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    split_by_id = {row["sample_id"]: row["split"] for row in splits}
    assert Counter(pair["primitive_key"] for pair in pairs) == Counter({value: 125 for value in ("attribute", "numerosity", "spatial", "occlusion")})
    for pair in pairs:
        members = by_pair[pair["pair_id"]]
        assert len(members) == 2 and members[0]["question"] == members[1]["question"]
        assert members[0]["answer"] != members[1]["answer"]
        assert {member["metadata"]["clean_or_corrupt"] for member in members} == {"clean", "corrupt"}
        assert split_by_id[members[0]["sample_id"]] == split_by_id[members[1]["sample_id"]]
        if pair["primitive_key"] == "attribute":
            clean_objects, corrupt_objects = (member["metadata"]["objects"] for member in members)
            assert len(clean_objects) == len(corrupt_objects) == 8
            assert Counter(obj["color"] for obj in clean_objects) == Counter({color: 2 for color in COLOR_OPTIONS})
            assert Counter(obj["color"] for obj in corrupt_objects) == Counter({color: 2 for color in COLOR_OPTIONS})
            assert Counter(obj["shape"] for obj in clean_objects) == Counter({shape: 2 for shape in SHAPES})
            assert Counter(obj["shape"] for obj in corrupt_objects) == Counter({shape: 2 for shape in SHAPES})
            assert sum(obj["outlined"] for obj in clean_objects) == sum(obj["outlined"] for obj in corrupt_objects) == 1
            assert all(
                {key: value for key, value in clean.items() if key != "color"}
                == {key: value for key, value in corrupt.items() if key != "color"}
                for clean, corrupt in zip(clean_objects, corrupt_objects)
            )
            changed = [
                object_index
                for object_index, (clean, corrupt) in enumerate(zip(clean_objects, corrupt_objects))
                if clean["color"] != corrupt["color"]
            ]
            assert changed == members[0]["metadata"]["color_swap_indices"]
            assert clean_objects[changed[0]]["shape"] == clean_objects[changed[1]]["shape"]
            assert clean_objects[changed[0]]["color"] == corrupt_objects[changed[1]]["color"]
            assert clean_objects[changed[1]]["color"] == corrupt_objects[changed[0]]["color"]
        if pair["primitive_key"] == "spatial":
            for member in members:
                for box in (member["metadata"]["subject_bbox"], member["metadata"]["reference_bbox"]):
                    assert 0 <= box[0] - 6 and box[2] + 6 < 448 and 0 <= box[1] - 6 and box[3] + 6 < 360, (member["sample_id"], box)
    for primitive, expected in {
        "attribute": sorted((62, 62, 63, 63)), "numerosity_individuation": [125, 125],
        "spatial_relation": sorted((62, 62, 63, 63)), "occlusion_object_unity": [125, 125],
    }.items():
        counts = sorted(Counter(row["answer"] for row in rows if row["primitive"] == primitive).values())
        assert counts == expected


if __name__ == "__main__":
    main()
