"""Prepare Qwen3.5-9B FP8_DYNAMIC, then run all frozen paper studies."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from _run_formal_model import run_fp8_dynamic
from compress_fp8_models import MODELS, compress_model


if __name__ == "__main__":
    compress_model(*MODELS[0])
    run_fp8_dynamic("qwen9b")
