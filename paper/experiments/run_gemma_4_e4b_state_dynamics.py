"""Run the frozen answer-state dynamics stages for google/gemma-4-E4B-it."""

from _run_formal_model import run_stages


STAGES = ("native_thinking_generation", "decodability_usability_gap")


if __name__ == "__main__":
    run_stages("gemma", STAGES)
