# From Perception to Integration

<p align="center">
  <a href="https://arxiv.org/abs/2609.34809"><img alt="arXiv: 2609.34809" src="https://img.shields.io/badge/arXiv-2609.34809-b31b1b" /></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue" /></a>
</p>

Code for **[From Perception to Integration: Revisiting the Internal Dynamics of Reasoning in Vision-Language Models](https://arxiv.org/abs/2609.34809)** by Rong Yu Xu, Prayag Tiwari, and Shaolei Zhang<sup>*</sup>.

<sup>*</sup> Corresponding author.

We study how vision-language models represent and combine visual judgments, and when their reasoning has progressed far enough to answer. This repository contains the dataset builders, behavioral and mechanistic experiments, and answer-readiness-guided early stopping code used in the paper.

## Repository contents

| Path | Contents |
|---|---|
| `datasets/` | Builders for controlled visual tasks and benchmark inputs |
| `src/vlm_core/` | Shared model, data, and scoring code |
| `paper/experiments/` | Experiment entry points and model launchers |
| `scripts/` | FP8 preparation, vLLM patch, and answer scoring |

The repository does not include model weights, generated datasets, saved predictions, trained detectors, or manuscript figures. Builders write data to `paper/data/`; experiments write results to `paper/results/`.

## Setup

Use a Linux machine with an NVIDIA GPU for model experiments. Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv venv --python 3.12
. .venv/bin/activate
uv pip install --index-strategy unsafe-best-match -r requirements.txt --overrides requirements-overrides.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
python scripts/apply_vllm_gemma4_pth_patch.py
```

The dependency files pin the tested runtime. Some model weights require Hugging Face access. FP8 checkpoints must be prepared with `scripts/compress_fp8_models.py` before running the FP8 models. Gemma-4-12B generation also uses the separately pinned SGLang backend in `requirements-sglang.txt`; it is not part of the base install.

## Run an experiment

Build the input datasets required by the experiment, for example:

```bash
python datasets/attribute_color_single.py
python datasets/numerosity_single.py
python datasets/spatial_relation_single.py
python datasets/amodal_shape_counterfactual_pairs.py
```

Then run the corresponding zero-argument entry point. The main experiment families are:

| Study | Entry point |
|---|---|
| Component behavior | `paper/experiments/behavioral_validation.py` |
| Hidden-state readouts | `paper/experiments/representation_decodability.py` |
| Causal patching | `paper/experiments/counterfactual_activation_patching.py` |
| Composite task | `paper/experiments/occluded_target_reasoning.py` |
| Answer-state dynamics | `paper/experiments/native_thinking_generation.py`, then `decodability_usability_gap.py` |
| MMStar / RealWorldQA early stopping | `paper/experiments/run_mmstar_external_early_stopping_qwen3_5_4b.py` and `run_realworldqa_external_early_stopping_qwen3_5_4b.py` (other model launchers are alongside them) |

The benchmark builders are `datasets/mmstar_official.py`, `datasets/realworldqa_official.py`, and `datasets/realworldqa_early_stopping_folds.py`. Early stopping also needs the answer-state analysis outputs and its model-specific launcher. Full GPU reruns have not been verified from this standalone repository.

## Citation

```bibtex
@misc{xu2026perceptionintegration,
  title = {From Perception to Integration: Revisiting the Internal Dynamics of Reasoning in Vision-Language Models},
  author = {Xu, Rong Yu and Tiwari, Prayag and Zhang, Shaolei},
  year = {2026},
  eprint = {2609.34809},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2609.34809}
}
```

## License

The original code is available under the [MIT License](LICENSE). The patch in `scripts/patches/` targets [vLLM](https://github.com/vllm-project/vllm), which is licensed under [Apache-2.0](licenses/Apache-2.0.txt). Models, benchmarks, and other dependencies retain their own licenses.
