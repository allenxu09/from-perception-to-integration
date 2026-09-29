"""Reference-independent final-choice extraction for saved benchmark outputs."""

import re

from vlm_core.realworldqa_answers import clean_model_markers

SCORING_VERSION = "benchmark_answer_scoring_v1"


def choice_answer(text: str, question: str = "") -> tuple[str, str]:
    """Prefer explicit answer assertions; never choose a later option mention.

    Inspect answer segments from last to first, retaining an earlier answer when
    repeated close markers leave only explanation. Option lists and unresolved
    alternatives are not affirmative selections. Gold is not an input.
    """
    segments = re.split(r"</think>", text, flags=re.I)
    if len(segments) == 1:
        direct = clean_model_markers(text).strip().strip("*_` .\n")
        return (direct, "direct_letter") if re.fullmatch(r"[A-D]", direct) else ("", "no_final_channel")
    options = dict(re.findall(r"(?m)^\s*\(?([A-D])\)?[.:]\s*(.+?)\s*$", question))
    for segment in reversed(segments[1:]):
        raw = clean_model_markers(segment).strip()
        final = raw.replace("**", "").replace("`", "")
        candidates = []
        patterns = (
            r"(?:\b(?:the\s+)?(?:final\s+|correct\s+)?answer\s*(?:is\s*|[:：]\s*)|\b(?:the\s+)?(?:correct|final)\s+(?:option|choice)\s*(?:is\s*|[:：]\s*))\s*(?:option\s+)?[\"'(]?([A-D])[\"')]?(?=$|[\s.:*])",
            r"\b(?:(?:I|we)\s+(?:choose|select)|(?:the\s+)?(?:correct\s+)?(?:option|choice)\s+is)\s*[:：]?\s*(?:option\s+)?\(?([A-D])\)?(?=$|[\s.,:])",
            r"\b(?:option\s+|choice\s+)?([A-D])\s+is\s+(?:the\s+)?(?:correct|intended|best)\s+(?:answer|choice|option|code)\b",
            r"\b(?:correct|final)\s+(?:answer|trade|choice|option)\s+is\s+option\s+([A-D])\b",
            r"(?m)^\s*(?:option|choice)\s*:\s*([A-D])\b",
            r"(?m)^\s*(?:Option|Choice)\s+\(?([A-D])\)?[.]?\s*$",
            r"\b(?:matches|corresponds to|corresponding to|closest to|is)\s+(?:option|choice)\s*\(?([A-D])\)?(?=$|[\s.,:])",
            r"\b(?:correct statement|best fit|most accurate (?:inference|relative position among the choices provided)|formula that has the same calculation result)\s+is\s*\(?([A-D])\)?(?=$|[\s.,:])",
            r"\\boxed\{\s*([A-D])\s*\}",
            r"<answer>\s*([A-D])\s*</answer>",
            r"(?m)^\s*[\"'(]?([A-D])[\"')]?[.)]?\s*_*\s*$",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, final, re.I):
                value = match.group(1)
                # Case-sensitive labels avoid reading the article 'a' as A.
                if value not in "ABCD":
                    continue
                tail = final[match.end():]
                if re.match(r"\s*(?:or\b|and\b|[,/]\s*\(?[A-D]\)?(?=[\s.,:]|$)|is\s+(?:not|incorrect|wrong)\b)", tail, re.I):
                    continue
                candidates.append((match.start(), value))
        if candidates:
            return max(candidates)[1], "explicit_answer"
        # A single leading labelled option followed by explanation is a reply;
        # multiple labelled lines are an option listing, not a choice.
        labelled = list(re.finditer(r"(?m)^\s*(?:Option\s+)?\(?([A-D])(?:\)[.:]?|[.:])\s+", final))
        if labelled and len({m.group(1) for m in labelled}) == 1:
            line = final[labelled[-1].end():].splitlines()[0]
            if not re.search(r"\b(?:incorrect|wrong)\b|\(false\)", line, re.I):
                return labelled[-1].group(1), "standalone_option"
        parenthetical = list(re.finditer(r"\(Option ([A-D])\)", final))
        if len(parenthetical) == 1 and not re.search(r"\b(?:incorrect|wrong|not|other options)\b", final, re.I):
            return parenthetical[0].group(1), "single_parenthetical_choice"
        bold = list(re.finditer(r"\*\*([A-D])\s*:", raw))
        if bold and len({m.group(1) for m in bold}) == 1:
            match = bold[-1]
            prefix = raw[max(0, match.start() - 90):match.start()]
            # Accept affirmative predicates, not bullets or option discussions.
            if re.search(r"\b(?:is|are|would be)\s*$", prefix, re.I) and not re.search(r"\b(?:not|maybe|likely|perhaps)\b", prefix, re.I):
                return match.group(1), "embedded_choice"
        normalized = re.sub(r"[^A-Z0-9]+", " ", final.upper()).strip()
        normalized = re.sub(r"^(?:THEREFORE )?(?:THE )?(?:FINAL |CORRECT )?ANSWER (?:IS )?", "", normalized)
        exact = [k for k, v in options.items() if normalized == re.sub(r"[^A-Z0-9]+", " ", v.upper()).strip()]
        if len(exact) == 1:
            return exact[0], "exact_option_text"
        if re.search(r"\b(?:cannot (?:decide|choose|answer)|withdraw|no final answer)\b", final, re.I):
            return "", "unresolved_final"
    return "", "no_explicit_choice"
