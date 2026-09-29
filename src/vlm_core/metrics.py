"""Small metrics used across experiments."""

from __future__ import annotations

import json
import re


def accuracy(predictions: list[str], answers: list[str]) -> float:
    if len(predictions) != len(answers):
        raise ValueError("predictions and answers must have the same length")
    if not answers:
        return 0.0
    correct = sum(pred.strip().lower() == gold.strip().lower() for pred, gold in zip(predictions, answers))
    return correct / len(answers)


def flip_rate(before: list[str], after: list[str]) -> float:
    if len(before) != len(after):
        raise ValueError("before and after must have the same length")
    if not before:
        return 0.0
    flips = sum(left.strip().lower() != right.strip().lower() for left, right in zip(before, after))
    return flips / len(before)


def overlap_ratio(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def exact_match(prediction: str, answer: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(answer)


def normalize_answer(text: str) -> str:
    return text.strip().lower().strip(".。")


def normalize_choice_answer(text: str, choices: list[str]) -> str:
    parsed = extract_structured_answer(text)
    normalized = normalize_answer(parsed if parsed is not None else text)
    if "</think>" in normalized:
        normalized = normalized.rsplit("</think>", 1)[-1].strip()
    if choices:
        choice_map = {normalize_answer(choice): choice for choice in choices}
        if normalized in choice_map:
            return normalized
        hits = [
            (normalized.rfind(choice), len(choice), choice)
            for choice in choice_map
            if choice and _choice_in_text(choice, normalized)
        ]
        if hits:
            return max(hits)[2]
    return normalized.splitlines()[-1].strip() if normalized.splitlines() else normalized


def _choice_in_text(choice: str, text: str) -> bool:
    if len(choice) == 1 and choice.isalpha():
        return re.search(rf"\b{re.escape(choice)}\b", text) is not None
    return choice in text


def extract_structured_answer(text: str) -> str | None:
    match = re.search(r"\{.*?}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "answer" in payload:
            return str(payload["answer"])

    boxed = re.search(r"\\boxed\{([^{}]+)}", text)
    if boxed:
        return boxed.group(1)
    return None
