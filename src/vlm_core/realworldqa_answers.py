"""Gold-independent, offline extraction of RealWorldQA short final answers."""

from __future__ import annotations

import re


SCORING_VERSION = "realworldqa_answer_scoring_v1"
_ATOM = r"(?:[+-]?\d+(?:\.\d+)?|[A-Za-z]+(?:[-'][A-Za-z]+)*)"
_LABEL = r"(?:(?:therefore,?\s+)?(?:the\s+)?(?:final\s+|correct\s+)?answer\s*(?:is\s*)?[:：]?\s*)"
_HEDGE = re.compile(r"\b(?:likely|maybe|perhaps|probably|possibly|assuming|unsure|uncertain)\b", re.I)


def clean_model_markers(text: str) -> str:
    return re.sub(r"<\|[^>]+\|>|<turn\|>|<channel\|>|<\|channel>(?:final|thought)", "\n", text, flags=re.I)


def extract_direct_answer(text: str, is_mc: bool = False) -> str:
    """Recognize a whole bare atomic reply, never unfinished reasoning prose."""
    cleaned = clean_model_markers(text).strip().replace("**", "").replace("`", "")
    cleaned = re.sub(rf"[\"'“‘]({_ATOM}[.!。]?)[\"'”’]", r"\1", cleaned)
    if not re.fullmatch(rf"{_ATOM}[.!。]?", cleaned):
        return ""
    return extract_final_answer(cleaned, is_mc=is_mc, require_closed=False)


def extract_choice_answer(text: str, question: str) -> str:
    """Read explicit choices, not mentions in prose or reproduced option lists."""
    if "</THINK>" not in text.upper():
        return ""
    final = clean_model_markers(text.upper().rsplit("</THINK>", 1)[-1])
    final = final.replace("**", "").replace("`", "")
    patterns = (
        r"<ANSWER>\s*\(?([A-D])\)?(?=\s*</ANSWER>)",
        r"(?m)^\s*(?:THEREFORE,?\s+)?(?:THE\s+)?(?:FINAL\s+|CORRECT\s+)?(?:ANSWER|OPTION|CHOICE)\s*(?:IS\s*)?[:\-]?\s*[\"'(]?([A-D])[\"')]?(?=$|[.\s])",
        r"(?m)^\s*[\"'(]?([A-D])[\"')]?[.)]?\s*$",
        r"\\BOXED\{\s*([A-D])\s*\}",
    )
    matches = [match for pattern in patterns for match in re.finditer(pattern, final)]
    matches = [match for match in matches if not re.match(r"\s*(?:OR\b|AND\b|[,/])", final[match.end():])]
    if matches:
        return max(matches, key=lambda match: match.start()).group(1)
    normalized = re.sub(r"[^A-Z0-9]+", " ", re.sub(rf"^{_LABEL}", "", final.strip(), flags=re.I)).strip()
    candidates = []
    for letter, option in re.findall(r"(?m)^([A-D])[.:]\s*(.+?)\s*$", question):
        option = re.sub(r"[^A-Z0-9]+", " ", option.upper()).strip()
        if option and normalized == option:
            candidates.append(letter)
    return candidates[0] if len(candidates) == 1 else ""


def extract_final_answer(text: str, is_mc: bool = False, require_closed: bool = True) -> str:
    """Extract an unambiguous atomic reply, including repeated identical replies.

    Only final-channel text is inspected. Explicit answer labels, boxed replies,
    standalone lines, and an initial atomic answer sentence are recognized.
    Explanatory prose is not interpreted; conflicting explicit answers and an
    initially qualified reply without an explicit final answer are rejected.
    No expected answers are consulted.
    """
    closes = list(re.finditer(r"</think>", text, re.I))
    if require_closed and not closes:
        return ""
    final = text[closes[-1].end():] if closes else text
    final = clean_model_markers(final).strip()
    final = final.replace("**", "").replace("`", "")
    final = re.sub(rf"[\"'“‘]({_ATOM}[.!。]?)[\"'”’]", r"\1", final)
    first_line = final.splitlines()[0] if final else ""

    candidates: list[str] = []

    def add(value: str) -> None:
        if re.fullmatch(_ATOM, value) and not _HEDGE.fullmatch(value):
            candidates.append(value.upper())

    for value in re.findall(r"\\boxed\{\s*([^{}]+?)\s*\}", final):
        add(value)
    atom = rf"[\"'‘“(]?({_ATOM})[\"'’”)]?"
    for line in final.splitlines():
        line = line.strip()
        clauses = re.split(r"(?<=[.!。])\s+", line)
        if len(clauses) > 1 and all(re.fullmatch(rf"{atom}[.!。]?", part) for part in clauses):
            for part in clauses:
                add(re.fullmatch(rf"{atom}[.!。]?", part).group(1))
        bare = re.fullmatch(rf"{atom}[.!。]?", line)
        labeled = re.fullmatch(rf"{_LABEL}{atom}[.!。]?", line, re.I)
        match = labeled or bare
        if match:
            add(match.group(1))

    initial = re.sub(rf"^{_LABEL}", "", final, count=1, flags=re.I)
    # A complete first answer sentence may precede an explanation. For numbers,
    # require a line break after the period to avoid reading a numbered step.
    match = re.match(rf"^{atom}(?P<boundary>[.!。](?=\s|$)|\n|$)", initial)
    if match and not _HEDGE.search(first_line):
        value = match.group(1)
        rest = initial[match.end():]
        if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value) or not rest.strip() or rest.startswith("\n"):
            add(value)
    unique = set(candidates)
    if len(unique) != 1:
        return ""
    answer = unique.pop()
    return answer if not is_mc or answer in {"A", "B", "C", "D"} else ""
