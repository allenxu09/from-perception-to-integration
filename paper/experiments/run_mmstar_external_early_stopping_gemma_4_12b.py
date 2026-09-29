"""Run formal MMStar external early stopping for Gemma 4 12B FP8_DYNAMIC."""

import importlib
import os


def main() -> None:
    os.environ["VLM_CORE_MMSTAR_EARLY_STOP_MODEL"] = "gemma12b"
    os.environ["VLM_CORE_SGLANG_AOT_FP8"] = "1"
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        value for value in (experiment_dir, os.environ.get("PYTHONPATH")) if value
    )
    importlib.import_module("answer_readiness_early_stopping").main()


if __name__ == "__main__":
    main()
