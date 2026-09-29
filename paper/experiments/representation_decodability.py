"""Frozen representation-decoding experiment."""

from _model_adapter import ModelAdapter
from _shared import ROOT, extract_features, load_samples, probe_rows, split_map, write_rows
from vlm_core.io import read_json, write_json

RIDGE = 100.0
SEED = 13

DATASETS = (
    ("attribute_color", ROOT / "paper/data/attribute_color_single", "target_color"),
    ("numerosity", ROOT / "paper/data/numerosity_single", "larger_side"),
    ("spatial_relation", ROOT / "paper/data/spatial_relation_single", "true_relation_label"),
    ("occlusion_unity", ROOT / "paper/data/occlusion_v2_amodal_fixed_1000", "complete_shape"),
)


def main(model_key: str = "qwen") -> None:
    adapter = ModelAdapter.load(model_key)
    batch_size = adapter.spec.batch_size_readout
    for task, data_dir, label_key in DATASETS:
        samples = load_samples(data_dir)
        splits = split_map(data_dir)
        labels = [sample.metadata[label_key] for sample in samples]
        manifest = read_json(data_dir / "manifest.json")
        result_dir = adapter.result_root / data_dir.name / "representation_decodability"
        cache_key = {
            "model": adapter.spec.model_id,
            "model_revision": adapter.spec.revision,
            "dataset_version": manifest["version"],
            "dataset_fingerprint": manifest["fingerprint"],
            "prompt_protocol": "no_think_json_v1",
            "batch_size": batch_size,
        }
        visual = extract_features(
            adapter,
            samples,
            batch_size,
            text_only=False,
            cache_dir=result_dir / "feature_shards_visual",
            cache_key={**cache_key, "branch": "image_and_question"},
        )
        text = extract_features(
            adapter,
            samples,
            batch_size,
            text_only=True,
            cache_dir=result_dir / "feature_shards_text",
            cache_key={**cache_key, "branch": "question_only"},
        )
        rows = probe_rows(visual, samples, labels, splits, task, "all", RIDGE, SEED)
        rows += probe_rows(
            text,
            samples,
            labels,
            splits,
            task,
            "all",
            RIDGE,
            SEED,
            controls=("question_only",),
        )

        if task == "numerosity":
            for ratio in (2.0, 1.5, 1.25):
                for area in ("congruent", "matched", "incongruent"):
                    rows += subset_probe(
                        visual,
                        samples,
                        splits,
                        label_key,
                        task,
                        f"ratio_{ratio:g}_{area}",
                        lambda sample, r=ratio, a=area: sample.metadata["ratio"] == r
                        and sample.metadata["area_condition"] == a,
                    )
        elif task == "spatial_relation":
            for distance in (60, 90, 120):
                for count in (4, 6, 8):
                    rows += subset_probe(
                        visual,
                        samples,
                        splits,
                        label_key,
                        task,
                        f"distance_{distance}_objects_{count}",
                        lambda sample, d=distance, n=count: sample.metadata["distance"] == d
                        and sample.metadata["n_objects"] == n,
                    )

        write_rows(result_dir, "probe_metrics", rows)
        write_json(
            result_dir / "provenance.json",
            {
                **adapter.provenance(),
                "dataset": data_dir.name,
                "dataset_version": manifest["version"],
                "dataset_fingerprint": manifest["fingerprint"],
                "samples": len(samples),
                "batch_size": batch_size,
                "ridge": RIDGE,
                "seed": SEED,
                "selected_layer": None,
                "layer_rule": "full hidden-state scan; no layer selected by protocol",
                "generation_config": "not_applicable",
                "feature_shards_visual": str(result_dir / "feature_shards_visual"),
                "feature_shards_text": str(result_dir / "feature_shards_text"),
                "output_path": str(result_dir / "probe_metrics.json"),
            },
        )


def subset_probe(features, samples, splits, label_key, task, name, keep):
    indices = [index for index, sample in enumerate(samples) if keep(sample)]
    subset_samples = [samples[index] for index in indices]
    subset_features = {group: [layer[indices] for layer in layers] for group, layers in features.items()}
    labels = [sample.metadata[label_key] for sample in subset_samples]
    return probe_rows(
        subset_features,
        subset_samples,
        labels,
        splits,
        task,
        name,
        RIDGE,
        SEED,
        controls=("main",),
    )


if __name__ == "__main__":
    main()
