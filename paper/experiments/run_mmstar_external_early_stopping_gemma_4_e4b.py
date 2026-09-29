"""Run formal MMStar external early stopping for Gemma 4 E4B."""

import importlib
import os


def main() -> None:
    os.environ["VLM_CORE_MMSTAR_EARLY_STOP_MODEL"] = "gemma"
    importlib.import_module("answer_readiness_early_stopping").main()


if __name__ == "__main__":
    main()
