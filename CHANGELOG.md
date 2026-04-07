# Changelog

All notable changes to the `egora` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] — 2026-04-07

### Fixed
- **Adaptive alpha now actually works**: `update_alpha()` is called automatically after each evaluation via `on_evaluate` callback. Previously it was never triggered.
- **All GovernorConfig parameters** (`alpha_min`, `alpha_max`, `alpha_gamma`, `gap_ema_beta`, `alpha_warmup`, `gap_spike_thresh`, `alpha_cooldown`, `alpha_cooldown_decay`, `alpha_cooldown_patience`) are now exposed in `EgoRALoraConfig` and `egora train` CLI.
- **Auto-enable eval_strategy**: When `adaptive_alpha=True` and `eval_dataset` is provided, evaluation strategy is automatically set to `"epoch"` if it was `"no"`.
- **CLI flags**: Added `--egora-alpha`, `--egora-lam-floor`, `--adaptive-alpha`, `--alpha-min`, `--alpha-max`, `--alpha-gamma`, `--alpha-cooldown` to `egora train`.

## [0.5.0] — 2026-04-07

### Added
- **`egora train` CLI**: One-command fine-tuning with automatic dataset loading, tokenization, EgoRA training, post-training diagnostics, and optional model merging. Example: `egora train meta-llama/Llama-3.2-1B tatsu-lab/alpaca --epochs 1 --rank 16 --save`.
- **`egora demo` CLI**: Interactive Gradio web dashboard for rotation diagnostics and model info. Launch with `egora demo` or `egora demo --share` for a public link. Requires `pip install gradio`.

## [0.4.0] — 2026-04-07

### Added
- **`EgoRATrainer`**: Drop-in HuggingFace `Trainer` replacement with automatic EgoRA adapter injection, entropy-governed loss, shadow refresh, and built-in diagnostics. Requires `transformers>=4.30`.
- **`EgoRATrainingArguments`**: Convenience dataclass with LoRA-tuned defaults.
- **Colab quickstart notebook** (`notebooks/quickstart_colab.ipynb`): Zero-install demo showing all three API levels (Trainer, PEFT-compat, low-level).

## [0.3.1] — 2026-04-07

### Fixed
- **Removed all arXiv references**: arXiv ID was incorrect; all citations now use Zenodo DOI ([10.5281/zenodo.19410504](https://doi.org/10.5281/zenodo.19410504)).
- **Zenodo DOI updated**: Corrected from `19398709` to `19410504` in all packages.

## [0.3.0] — 2026-04-07

### Fixed
- **GitHub URLs**: All three packages (`egora`, `egora-diagnostics`, `egora-rrl`) now correctly point to `https://github.com/ArsSocratica/EgoRA`.
- **LICENSE-ACADEMIC**: Citation requirement updated with correct Zenodo DOI in all packages.
- **Sub-package READMEs**: Replaced placeholder text with full documentation including installation, usage examples, and correct AGPL-3.0 license info (was incorrectly listed as "Apache 2.0").

## [0.2.1] — 2026-04-03

### Fixed
- README links (LICENSE-ACADEMIC, examples/) now use absolute GitHub URLs so they work on PyPI.

## [0.2.0] — 2026-04-03

### Added
- **CLI tool** (`egora diagnose`, `egora info`, `egora version`): Command-line interface for rotation diagnostics, model info, and version checking.
- **Plotting module** (`egora.plotting`): Publication-quality visualizations including rotation heatmaps, layer profiles, learning mode distribution, projection comparisons, and combined 4-panel reports.
- **PEFT compatibility layer** (`egora.peft_compat`): `EgoRAPeftModel` wrapper with PEFT-like API (`from_pretrained`, `save_pretrained`, `merge_and_unload`, `print_trainable_parameters`). `EgoRALoraConfig` for JSON-serializable configuration.
- **Example scripts** (`examples/`): `quickstart_training.py` (end-to-end fine-tuning), `diagnostics_only.py` (standalone analysis), `peft_compat_example.py` (PEFT-style API demo).
- 24 new tests (CLI, plotting, PEFT compat). Total: 51 tests.

## [0.1.0] — 2026-04-03

### Added
- **EgoRA-LoRA adapter** (`EgoRALoRALinear`): Standard LoRA with entropy-governed orthogonality penalty.
- **DoRA adapter** (`DoRALinear`): Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024).
- **rsLoRA adapter** (`rsLoRALinear`): Rank-Stabilized LoRA (Kalajdzievski, 2023).
- **Entropy Governor** (`EntropyGovernor`): Dynamic lambda controller with optional adaptive alpha.
- **Geometry diagnostics** (`compute_head_geometry`, `compute_interhead_diversity`): Per-head rotation analysis, learning mode classification, and diversity metrics.
- **Utility functions**: `apply_lora`, `get_total_egora_penalty`, `refresh_all_shadows`, `merge_lora`, `unmerge_lora`.
- **`save_geometry`**: Export rotation results to JSON + numpy arrays.
- Unit tests for adapters, governor, and geometry modules.
- AGPL-3.0 license with Academic Additional Permission (Section 7).
- Patent notice for U.S. Provisional No. 64/024,742.
