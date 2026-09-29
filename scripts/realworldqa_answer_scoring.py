"""Score completed RealWorldQA predictions without model inference or training."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from vlm_core.io import read_jsonl, write_json, write_jsonl
from vlm_core.realworldqa_answers import (
    SCORING_VERSION,
    extract_choice_answer,
    extract_direct_answer,
    extract_final_answer,
)

# Frozen offline scoring scope. MMStar is deliberately absent.
ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "qwen3_5_4b",
    "gemma_4_e4b_it",
    "qwen3_5_9b_fp8_dynamic",
    "gemma_4_12b_it_fp8_dynamic",
)
DIRECTIONS = {"A_to_B": 382, "B_to_A": 383}
MANIFEST = ROOT / "paper/data/realworldqa/manifest.jsonl"


def metrics(rows):
    n = len(rows)
    return {
        "n": n,
        "correct": sum(row["is_correct"] for row in rows),
        "accuracy": sum(row["is_correct"] for row in rows) / n,
        "mean_reasoning_tokens": sum(row["reasoning_tokens"] for row in rows) / n,
        "score_changes": dict(Counter(
            f"{int(row['original_is_correct'])}->{int(row['is_correct'])}"
            for row in rows if row["original_is_correct"] != row["is_correct"]
        )),
    }


def main():
    questions = {int(row["index"]): row for row in read_jsonl(MANIFEST)}
    for model in MODELS:
        source = ROOT / "paper/results" / model / "realworldqa_external_early_stopping"
        files = {(direction, condition): source / direction / f"{condition}_predictions.jsonl"
                 for direction in DIRECTIONS for condition in ("full", "adaptive")}
        if not all(path.exists() for path in files.values()):
            print(f"{model}: predictions not complete; no scores written")
            continue
        raw = {
            key: list({int(row["case_id"]): row for row in read_jsonl(path)}.values())
            for key, path in files.items()
        }
        if any(len(rows) != DIRECTIONS[direction] for (direction, _), rows in raw.items()):
            print(f"{model}: predictions not complete; no scores written")
            continue
        output = source / "answer_scores"
        pooled = {condition: [] for condition in ("full", "adaptive")}
        summaries = {}
        for (direction, condition), rows in raw.items():
            assert len({row["case_id"] for row in rows}) == len(rows)
            scored = []
            for row in rows:
                question = questions[row["case_id"]]
                prediction = extract_choice_answer(row["prediction"], question["question"])
                if question["question_type"] == "short_answer":
                    prediction = extract_final_answer(row["prediction"], is_mc=False, require_closed=True)
                direct = ""
                if "</think>" not in row["prediction"].lower():
                    direct = extract_direct_answer(row["prediction"], is_mc=question["question_type"] == "multiple_choice")
                    prediction = direct
                scored.append({
                    **row,
                    "original_extracted_choice": row["extracted_choice"],
                    "original_is_correct": row["is_correct"],
                    "original_reasoning_tokens": row["reasoning_tokens"],
                    "original_early_stopped": row["early_stopped"],
                    "original_forced_stop_checkpoint": row["forced_stop_checkpoint"],
                    "direct_answer_without_thinking": bool(direct),
                    "reasoning_tokens": 0 if direct else row["reasoning_tokens"],
                    "early_stopped": False if direct else row["early_stopped"],
                    "forced_stop_checkpoint": None if direct else row["forced_stop_checkpoint"],
                    "extracted_choice": prediction,
                    "is_correct": prediction == str(question["answer"]).strip().upper(),
                    "scoring_version": SCORING_VERSION,
                })
            write_jsonl(output / direction / f"{condition}_predictions.jsonl", scored)
            summaries.setdefault(direction, {})[condition] = metrics(scored)
            pooled[condition].extend(scored)
        aggregate = {condition: metrics(rows) for condition, rows in pooled.items()}
        summary = {
            "scoring_version": SCORING_VERSION,
            "scope": "fixed saved policy; no inference, label changes, retraining, or threshold reselection",
            "multiple_choice_parser": "last explicit choice; rejects option lists and unresolved alternatives; exact option-text fallback",
            "direct_answer_rule": "whole atomic reply with no thinking marker is a direct final; reasoning count zero; post-EOS stop metadata ignored",
            "directions": summaries,
            "pooled": aggregate,
            "adaptive_accuracy_change_pp": 100 * (aggregate["adaptive"]["accuracy"] - aggregate["full"]["accuracy"]),
            "reasoning_token_reduction": 1 - aggregate["adaptive"]["mean_reasoning_tokens"] / aggregate["full"]["mean_reasoning_tokens"],
        }
        write_json(output / "summary.json", summary)
        print(model, aggregate, flush=True)


if __name__ == "__main__":
    main()
