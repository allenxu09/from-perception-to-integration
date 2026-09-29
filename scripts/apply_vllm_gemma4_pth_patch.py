"""Apply the pinned Gemma 4 per-token-head KV-cache fix to the vLLM wheel."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path

VLLM_VERSION = "0.27.2rc1.dev77+gac7509e2b.cu129"
PR_HEAD = "c74e90b9e2f457306f18ba593d6303e35fb560ec"
PATCH = Path(__file__).with_name("patches") / "vllm_gemma4_pth_ac7509.patch"
REQUIRED = {
    "model_executor/models/gemma4.py": "kv_cache_page_size_padded=kv_cache_page_size_padded",
    "model_executor/layers/attention/attention.py": "self.kv_cache_page_size_padded",
    "v1/core/kv_cache_utils.py": (
        "block_size=new_block_size,\n"
        "                    page_size_padded=max_page_size"
    ),
    "v1/worker/gpu/attn_utils.py": "padded_last_dim",
}


def main() -> None:
    version = importlib.metadata.version("vllm")
    if version != VLLM_VERSION:
        raise RuntimeError(f"Expected vLLM {VLLM_VERSION}, found {version}")

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("vLLM is not installed")
    package = Path(spec.origin).resolve().parent

    if not is_applied(package):
        subprocess.run(
            [
                "patch",
                "--batch",
                "--forward",
                "-p1",
                "-d",
                str(package.parent),
                "-i",
                str(PATCH),
            ],
            check=False,
        )
    if not is_applied(package):
        raise RuntimeError("Gemma 4 vLLM patch did not apply cleanly")

    marker = {
        "base_vllm": VLLM_VERSION,
        "upstream_pr": 40391,
        "pr_head": PR_HEAD,
        "kv_cache_dtype": "fp8_per_token_head",
    }
    (package / "GEMMA4_PTH_PATCH.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    print(f"patched {package}")


def is_applied(package: Path) -> bool:
    return all(
        token in (package / relative).read_text(encoding="utf-8")
        for relative, token in REQUIRED.items()
    )


if __name__ == "__main__":
    main()
