"""Run the frozen paper studies for one fixed model in isolated processes."""

from __future__ import annotations

import importlib
import multiprocessing


STAGES = (
    "behavioral_validation",
    "representation_decodability",
    "counterfactual_activation_patching",
    "native_thinking_generation",
    "decodability_usability_gap",
)

FP8_DYNAMIC_STAGES = (
    "behavioral_validation",
    "representation_decodability",
    "counterfactual_activation_patching",
    "occluded_target_reasoning",
    "native_thinking_generation",
    "decodability_usability_gap",
)


def run(model_key: str) -> None:
    run_stages(model_key, STAGES)


def run_fp8_dynamic(model_key: str) -> None:
    run_stages(model_key, FP8_DYNAMIC_STAGES)


def run_stages(model_key: str, stages) -> None:
    context = multiprocessing.get_context("spawn")
    for module_name in stages:
        process = context.Process(target=run_stage, args=(module_name, model_key))
        process.start()
        process.join()
        if process.exitcode != 0:
            raise SystemExit(f"{module_name} failed for {model_key} with exit code {process.exitcode}")


def run_stage(module_name: str, model_key: str) -> None:
    importlib.import_module(module_name).main(model_key)
