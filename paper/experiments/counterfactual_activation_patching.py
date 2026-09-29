"""Frozen bidirectional counterfactual activation patching."""

import random
import statistics
from collections import Counter, defaultdict

import torch

from _model_adapter import ModelAdapter
from _shared import ROOT, load_samples, patch_prompt, token_masks, write_rows
from vlm_core.io import append_jsonl, read_json, read_jsonl, write_json, write_jsonl

SPLIT = "test"
TOKEN_GROUPS = ("image_tokens", "question_tokens", "last_prompt_token")
RANDOM_SEED = 20260808
SEMANTIC_KEYS = {
    "attribute": "target_color",
    "numerosity": "larger_side",
    "spatial_relation": "true_relation_label",
    "amodal_shape": "complete_shape",
}

PAIR_DATA = ROOT / "paper/data/single_primitive_counterfactual_pairs"
TASKS = (
    ("attribute", PAIR_DATA, "attribute", "attribute_outlined_target_cf_v2"),
    ("numerosity", PAIR_DATA, "numerosity_individuation", "nasco_dot_array_cf_v2"),
    ("spatial_relation", PAIR_DATA, "spatial_relation", "spatial_marked_relation_cf_v2"),
    (
        "amodal_shape",
        ROOT / "paper/data/occlusion_v2_amodal_fixed_1000",
        "occlusion_amodal_completion",
        "amodal_completion_v2",
    ),
)


def main(model_key: str = "qwen") -> None:
    adapter = ModelAdapter.load(model_key)
    layers = adapter.decoder_layers()
    batch_size = adapter.spec.batch_size_patching
    for task, data_dir, primitive, subtask in TASKS:
        task_batch_size = batch_size
        samples = [
            sample
            for sample in load_samples(data_dir, SPLIT)
            if sample.primitive == primitive and sample.subtask == subtask
        ]
        directions = strict_directions(samples)
        random_donors = matched_random_donors(
            samples,
            SEMANTIC_KEYS[task],
            adapter.tokenizer,
            match_question_length=task == "amodal_shape",
        )
        result_dir = adapter.result_root / "counterfactual_activation_patching" / task
        result_dir.mkdir(parents=True, exist_ok=True)
        provenance = run_provenance(adapter, task, data_dir, len(directions), task_batch_size)
        metric_path = result_dir / "pair_metrics.jsonl"
        check_resume(result_dir, metric_path, provenance)
        write_json(result_dir / "provenance.json", provenance)
        write_jsonl(
            result_dir / "directions.jsonl",
            [
                {
                    "pair_id": donor.counterfactual_id,
                    "donor": donor.sample_id,
                    "receiver": receiver.sample_id,
                    "matched_random_donor": random_donors[receiver.sample_id].sample_id,
                }
                for donor, receiver in directions
            ],
        )

        old_rows = read_jsonl(metric_path) if metric_path.exists() else []
        indexed = {(row["direction"], row["token_group"], row["layer"]): row for row in old_rows}
        expected = len(layers) * len(TOKEN_GROUPS)
        counts = Counter(key[0] for key in indexed)
        pending = [
            pair
            for pair in directions
            if counts[f"{pair[0].sample_id}->{pair[1].sample_id}"] != expected
        ]

        for start in range(0, len(pending), task_batch_size):
            batch_rows = patch_batch(
                adapter,
                layers,
                pending[start : start + task_batch_size],
                random_donors,
                task,
                SEMANTIC_KEYS[task],
            )
            append_jsonl(metric_path, batch_rows)
            for row in batch_rows:
                indexed[(row["direction"], row["token_group"], row["layer"])] = row

        pair_rows = list(indexed.values())
        if len(pair_rows) != len(directions) * len(layers) * len(TOKEN_GROUPS):
            raise SystemExit(f"Incomplete activation patching output: {result_dir}")
        write_jsonl(metric_path, pair_rows)
        write_rows(result_dir, "metrics", aggregate(pair_rows, task))


def strict_directions(samples):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.counterfactual_id].append(sample)
    pairs = []
    for pair_id, members in groups.items():
        clean = [sample for sample in members if sample.metadata["clean_or_corrupt"] == "clean"]
        corrupt = [sample for sample in members if sample.metadata["clean_or_corrupt"] == "corrupt"]
        if len(members) != 2 or len(clean) != 1 or len(corrupt) != 1:
            raise ValueError(f"Invalid pair {pair_id}")
        first, second = clean[0], corrupt[0]
        if first.question != second.question or first.choices != second.choices or first.answer == second.answer:
            raise ValueError(f"Unmatched pair {pair_id}")
        pairs.append((first, second))
    return [direction for first, second in pairs for direction in ((first, second), (second, first))]


def matched_random_donors(samples, semantic_key, tokenizer, match_question_length):
    groups = defaultdict(list)
    for sample in samples:
        key = (sample.metadata[semantic_key],)
        if match_question_length:
            key += (len(tokenizer(sample.question, add_special_tokens=False).input_ids),)
        groups[key].append(sample)

    rng = random.Random(RANDOM_SEED)
    donors = {}
    for key in sorted(groups, key=str):
        group = sorted(groups[key], key=lambda sample: sample.sample_id)
        rng.shuffle(group)
        for index, receiver in enumerate(group):
            candidates = group[index + 1 :] + group[:index]
            donor = next(
                (candidate for candidate in candidates if candidate.counterfactual_id != receiver.counterfactual_id),
                None,
            )
            if donor is None:
                raise ValueError(f"No matched-random donor for {receiver.sample_id}")
            donors[receiver.sample_id] = donor
    return donors


def run_provenance(adapter, task, data_dir, direction_count, batch_size):
    manifest = read_json(data_dir / "manifest.json")
    return {
        **adapter.provenance(),
        "task": task,
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["version"],
        "dataset_fingerprint": manifest["fingerprint"],
        "split": SPLIT,
        "pairs": direction_count // 2,
        "directions": direction_count,
        "batch_size": batch_size,
        "prompt_protocol": "no_think_answer_suffix_v1",
        "patch_component": "resid_pre",
        "regions": list(TOKEN_GROUPS),
        "bidirectional": True,
        "behavior_valid_primary_rule": "both pair members correct over complete choice set; include both directions",
        "matched_random_rule": "same primitive/test split/receiver semantic value; different pair; amodal also same question-token count",
        "matched_random_semantic_key": SEMANTIC_KEYS[task],
        "matched_random_seed": RANDOM_SEED,
        "recovery_definition": "(patched_margin - receiver_margin) / (donor_margin - receiver_margin)",
        "selected_layer": None,
        "layer_rule": "all decoder blocks",
        "generation_config": "not_applicable_next_token_scoring",
        "output_path": str(adapter.result_root / "counterfactual_activation_patching" / task / "pair_metrics.jsonl"),
    }


def check_resume(result_dir, metric_path, provenance):
    provenance_path = result_dir / "provenance.json"
    if metric_path.exists() and (not provenance_path.exists() or read_json(provenance_path) != provenance):
        raise RuntimeError(f"Existing results are incompatible with the frozen activation patching config: {result_dir}")


@torch.inference_mode()
def patch_batch(adapter, layers, directions, random_donors, task, semantic_key):
    donors = [donor for donor, _ in directions]
    receivers = [receiver for _, receiver in directions]
    controls = [random_donors[receiver.sample_id] for receiver in receivers]
    donor_inputs, donor_masks, donor_outputs = forward(adapter, donors, output_hidden_states=True)
    donor_margin = margin(adapter, donor_inputs, donor_outputs.logits, donors, receivers)
    donor_prediction = predictions(adapter, donor_inputs, donor_outputs.logits, donors)
    donor_states = [hidden.detach().to("cpu") for hidden in donor_outputs.hidden_states[:-1]]
    del donor_outputs
    torch.cuda.empty_cache()

    receiver_inputs, receiver_masks, receiver_outputs = forward(adapter, receivers, output_hidden_states=False)
    receiver_margin = margin(adapter, receiver_inputs, receiver_outputs.logits, donors, receivers)
    receiver_prediction = predictions(adapter, receiver_inputs, receiver_outputs.logits, receivers)
    del receiver_outputs
    torch.cuda.empty_cache()
    behavior_valid = {
        donor.counterfactual_id: donor_prediction[index] == donor.answer
        and receiver_prediction[index] == receiver.answer
        for index, (donor, receiver) in enumerate(directions)
    }
    _, control_masks, control_outputs = forward(adapter, controls, output_hidden_states=True)
    control_states = [hidden.detach().to("cpu") for hidden in control_outputs.hidden_states[:-1]]
    del control_outputs
    torch.cuda.empty_cache()
    denominator = donor_margin - receiver_margin
    rows = []

    for layer_index in range(len(layers)):
        for group in TOKEN_GROUPS:
            patched = adapter.patched_forward(
                receiver_inputs,
                layer_index,
                donor_states[layer_index],
                donor_masks[group],
                receiver_masks[group],
            )
            patched_margin = margin(adapter, receiver_inputs, patched.logits, donors, receivers)
            random_patched = adapter.patched_forward(
                receiver_inputs,
                layer_index,
                control_states[layer_index],
                control_masks[group],
                receiver_masks[group],
            )
            random_patched_margin = margin(adapter, receiver_inputs, random_patched.logits, donors, receivers)
            for index, (donor, receiver) in enumerate(directions):
                denom = denominator[index].item()
                recovery = (patched_margin[index].item() - receiver_margin[index].item()) / denom if abs(denom) > 1e-6 else None
                random_recovery = (
                    (random_patched_margin[index].item() - receiver_margin[index].item()) / denom
                    if abs(denom) > 1e-6
                    else None
                )
                control = controls[index]
                rows.append(
                    {
                        "task": task,
                        "pair_id": donor.counterfactual_id,
                        "direction": f"{donor.sample_id}->{receiver.sample_id}",
                        "token_group": group,
                        "layer": layer_index,
                        "donor_margin": donor_margin[index].item(),
                        "receiver_margin": receiver_margin[index].item(),
                        "patched_margin": patched_margin[index].item(),
                        "recovery": recovery,
                        "matched_random_donor": control.sample_id,
                        "matched_random_semantic_value": control.metadata[semantic_key],
                        "matched_random_patched_margin": random_patched_margin[index].item(),
                        "matched_random_recovery": random_recovery,
                        "donor_correct": donor_prediction[index] == donor.answer,
                        "receiver_correct": receiver_prediction[index] == receiver.answer,
                        "behavior_valid_pair": behavior_valid[donor.counterfactual_id],
                        "moves_toward_donor": patched_margin[index].item() > receiver_margin[index].item(),
                        "matched_random_moves_toward_donor": random_patched_margin[index].item()
                        > receiver_margin[index].item(),
                    }
                )
    return rows


def forward(adapter, samples, output_hidden_states):
    inputs = adapter.prepare_inputs(
        [patch_prompt(sample.question) for sample in samples],
        [sample.image_path for sample in samples],
        enable_thinking=False,
    )
    outputs = adapter.forward(inputs, output_hidden_states=output_hidden_states)
    return inputs, token_masks(inputs, adapter, samples, text_only=False), outputs


def last_logits(inputs, logits):
    if logits.shape[1] == 1:
        return logits[:, 0]
    positions = inputs["attention_mask"].shape[1] - 1 - inputs["attention_mask"].flip(1).int().argmax(1)
    return logits[torch.arange(len(logits), device=logits.device), positions]


def answer_score(tokenizer, logits, answer):
    ids = set()
    for text in (answer, f" {answer}", answer.lower(), f" {answer.lower()}"):
        encoded = tokenizer(text, add_special_tokens=False).input_ids
        if encoded:
            ids.add(encoded[-1])
    return logits[list(ids)].max()


def margin(adapter, inputs, logits, donors, receivers):
    final = last_logits(inputs, logits)
    return torch.stack(
        [
            answer_score(adapter.tokenizer, final[row], donor.answer)
            - answer_score(adapter.tokenizer, final[row], receiver.answer)
            for row, (donor, receiver) in enumerate(zip(donors, receivers))
        ]
    )


def predictions(adapter, inputs, logits, samples):
    final = last_logits(inputs, logits)
    return [
        max(sample.choices, key=lambda answer: answer_score(adapter.tokenizer, final[row], answer).item())
        for row, sample in enumerate(samples)
    ]


def aggregate(rows, task):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["token_group"], row["layer"])].append(row)
    output = []
    for (group, layer), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        secondary = [row for row in values if row["recovery"] is not None]
        pair_rows = defaultdict(list)
        for row in secondary:
            pair_rows[row["pair_id"]].append(row)
        primary_pair_ids = {
            pair_id
            for pair_id, pair in pair_rows.items()
            if len({row["direction"] for row in pair}) == 2 and all(row["behavior_valid_pair"] for row in pair)
        }
        primary = [row for row in secondary if row["pair_id"] in primary_pair_ids]
        true_primary = recovery_summary(primary, "recovery", "moves_toward_donor")
        random_primary = recovery_summary(
            primary,
            "matched_random_recovery",
            "matched_random_moves_toward_donor",
        )
        difference_primary = difference_summary(primary)
        true_secondary = recovery_summary(secondary, "recovery", "moves_toward_donor")
        random_secondary = recovery_summary(
            secondary,
            "matched_random_recovery",
            "matched_random_moves_toward_donor",
        )
        output.append(
            {
                "task": task,
                "token_group": group,
                "layer": layer,
                "primary_pairs": len(primary_pair_ids),
                "primary_directions": len(primary),
                "primary_mean_recovery": true_primary["mean"],
                "primary_median_recovery": true_primary["median"],
                "primary_bootstrap_low": true_primary["low"],
                "primary_bootstrap_high": true_primary["high"],
                "primary_moves_toward_donor_rate": true_primary["moves_rate"],
                "matched_random_primary_mean_recovery": random_primary["mean"],
                "matched_random_primary_median_recovery": random_primary["median"],
                "matched_random_primary_bootstrap_low": random_primary["low"],
                "matched_random_primary_bootstrap_high": random_primary["high"],
                "matched_random_primary_moves_toward_donor_rate": random_primary["moves_rate"],
                "primary_mean_recovery_difference": difference_primary["mean"],
                "primary_difference_bootstrap_low": difference_primary["low"],
                "primary_difference_bootstrap_high": difference_primary["high"],
                "secondary_valid_directions": len(secondary),
                "secondary_mean_recovery": true_secondary["mean"],
                "matched_random_secondary_mean_recovery": random_secondary["mean"],
                "secondary_mean_recovery_difference": difference_summary(secondary)["mean"],
            }
        )
    return output


def recovery_summary(rows, recovery_field, moves_field):
    values = [row[recovery_field] for row in rows]
    pair_values = pair_means(rows, recovery_field)
    low, high = bootstrap(pair_values)
    return {
        "mean": round(statistics.mean(values), 6) if values else None,
        "median": round(statistics.median(values), 6) if values else None,
        "low": low,
        "high": high,
        "moves_rate": round(sum(row[moves_field] for row in rows) / len(rows), 6) if rows else None,
    }


def difference_summary(rows):
    pair_values = pair_means(rows, "recovery", subtract="matched_random_recovery")
    low, high = bootstrap(pair_values)
    return {
        "mean": round(statistics.mean(pair_values), 6) if pair_values else None,
        "low": low,
        "high": high,
    }


def pair_means(rows, field, subtract=None):
    values = defaultdict(list)
    for row in rows:
        value = row[field] - row[subtract] if subtract else row[field]
        values[row["pair_id"]].append(value)
    return [statistics.mean(pair) for pair in values.values()]


def bootstrap(values):
    if not values:
        return None, None
    rng = random.Random(13)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(1000))
    return round(means[24], 6), round(means[974], 6)


if __name__ == "__main__":
    main()
