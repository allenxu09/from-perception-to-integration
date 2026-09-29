"""Run formal MMStar external early stopping for Qwen3.5-9B FP8_DYNAMIC."""

import importlib
import os


def main() -> None:
    os.environ["VLM_CORE_MMSTAR_EARLY_STOP_MODEL"] = "qwen9b"
    importlib.import_module("answer_readiness_early_stopping").main()


if __name__ == "__main__":
    main()
