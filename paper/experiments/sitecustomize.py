"""Allocation-specific startup fixes for experiment subprocesses."""

import os


if os.environ.get("VLM_CORE_SGLANG_AOT_FP8") == "1":
    from sgl_kernel import sgl_per_token_quant_fp8
    from sglang.kernels.ops.quantization import fp8_kernel

    fp8_kernel.sgl_per_token_quant_fp8 = sgl_per_token_quant_fp8
