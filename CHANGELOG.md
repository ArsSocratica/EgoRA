# Changelog

All notable changes to the `egora` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
