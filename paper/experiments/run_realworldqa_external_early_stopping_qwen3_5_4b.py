"""Run frozen RealWorldQA two-fold external early stopping for qwen3_5_4b."""

import importlib
import os


def main() -> None:
    os.environ["VLM_CORE_MMSTAR_EARLY_STOP_MODEL"] = "qwen"
    os.environ["VLM_CORE_EARLY_STOP_DATASET"] = "realworldqa"
    importlib.import_module("answer_readiness_early_stopping").main()


if __name__ == "__main__":
    main()
