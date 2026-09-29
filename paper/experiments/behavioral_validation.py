"""Frozen behavioral validation."""

from collections import defaultdict

from _model_adapter import generation_adapter
from _shared import ROOT, load_samples, prompt
from vlm_core.io import append_jsonl, read_json, read_jsonl, write_json
from vlm_core.metrics import exact_match, normalize_choice_answer
from vlm_core.runner import batched

MAX_NEW_TOKENS = 40
SEED = 13
PROMPT_PROTOCOL = "no_think_json_v1"

DATASETS = (
    ROOT / "paper/data/attribute_color_single",
    ROOT / "paper/data/numerosity_single",
    ROOT / "paper/data/spatial_relation_single",
    ROOT / "paper/data/occlusion_v2_amodal_fixed_1000",
)


def main(model_key: str = "qwen") -> None:
    adapter = generation_adapter(model_key)
    batch_size = adapter.spec.batch_size_behavioral
    for data_dir in DATASETS:
        samples = load_samples(data_dir)
        manifest = read_json(data_dir / "manifest.json")
        result_dir = adapter.result_root / "behavioral_validation" / data_dir.name
        prediction_path = result_dir / "predictions.jsonl"
        provenance_path = result_dir / "provenance.json"
        result_dir.mkdir(parents=True, exist_ok=True)
        provenance = {
            **adapter.provenance(),
            "dataset": data_dir.name,
            "dataset_version": manifest["version"],
            "dataset_fingerprint": manifest["fingerprint"],
            "samples": len(samples),
            "batch_size": batch_size,
            "generation_config": adapter.generation_provenance(False, MAX_NEW_TOKENS, SEED),
            "prompt_protocol": PROMPT_PROTOCOL,
            "output_path": str(prediction_path),
        }
        if prediction_path.exists() and (not provenance_path.exists() or read_json(provenance_path) != provenance):
            raise SystemExit(f"Existing behavioral validation results have incompatible provenance: {result_dir}")
        write_json(provenance_path, provenance)
        existing = read_jsonl(prediction_path) if prediction_path.exists() else []
        expected_ids = {sample.sample_id for sample in samples}
        existing_ids = [row["sample_id"] for row in existing]
        if len(existing_ids) != len(set(existing_ids)) or set(existing_ids) - expected_ids:
            raise SystemExit(f"Invalid behavioral validation checkpoint: {prediction_path}")
        rows = {row["sample_id"]: row for row in existing}
        pending = [sample for sample in samples if sample.sample_id not in rows]

        for batch_index, batch in enumerate(batched(pending, batch_size)):
            outputs = adapter.generate(
                [prompt(sample.question) for sample in batch],
                [sample.image_path for sample in batch],
                enable_thinking=False,
                max_new_tokens=MAX_NEW_TOKENS,
                seed=SEED + batch_index,
            )
            saved = []
            for output, sample in zip(outputs, batch):
                normalized = normalize_choice_answer(output["clean_text"], sample.choices)
                correct = exact_match(normalized, sample.answer) or exact_match(
                    normalized,
                    str(sample.metadata.get("answer_value", "")),
                )
                row = {
                    "sample_id": sample.sample_id,
                    "pair_id": sample.counterfactual_id,
                    "role": sample.metadata.get("clean_or_corrupt"),
                    "answer": sample.answer,
                    "prediction": normalized,
                    "raw_output": output["text"],
                    "correct": correct,
                }
                saved.append(row)
                rows[sample.sample_id] = row
            append_jsonl(prediction_path, saved)

        ordered = [rows[sample.sample_id] for sample in samples]
        by_primitive = defaultdict(list)
        by_pair = defaultdict(list)
        for sample, row in zip(samples, ordered):
            by_primitive[sample.primitive].append(row)
            if sample.counterfactual_id:
                by_pair[sample.counterfactual_id].append(row)
        write_json(
            result_dir / "metrics.json",
            {
                "overall": accuracy(ordered),
                "by_primitive": {name: accuracy(items) for name, items in by_primitive.items()},
                "pair_both_correct": pair_accuracy(by_pair),
                "valid_predictions": sum(bool(row["prediction"]) for row in ordered),
                "expected_samples": len(samples),
            },
        )


def accuracy(rows) -> float:
    return round(sum(row["correct"] for row in rows) / len(rows), 6)


def pair_accuracy(by_pair) -> float | None:
    pairs = [rows for rows in by_pair.values() if len(rows) == 2]
    return round(sum(all(row["correct"] for row in rows) for rows in pairs) / len(pairs), 6) if pairs else None


if __name__ == "__main__":
    main()
