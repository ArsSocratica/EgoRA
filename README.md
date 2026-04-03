# EgoRA — Entropy-Governed Orthogonality Regularization for Adaptation

[![PyPI version](https://img.shields.io/pypi/v/egora)](https://pypi.org/project/egora/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dataset on HF](https://img.shields.io/badge/🤗_Dataset-egora--benchmarks-yellow)](https://huggingface.co/datasets/ArsSocratica/egora-benchmarks)
[![arXiv](https://img.shields.io/badge/arXiv-2602.05192-b31b1b.svg)](https://arxiv.org/abs/2602.05192)

**EgoRA** is a dynamic, information-theoretic regularization method for fine-tuning neural networks. It uses the model's own output entropy to modulate an orthogonality penalty on LoRA adapter weights, preventing knowledge destruction and rank collapse during adaptation.

The method is based on the **Rotation-Retention Law**: knowledge loss in fine-tuned language models is proportional to representational rotation ($\Delta M \propto \bar{\theta}$).

## Installation

```bash
pip install egora
```

With diagnostics support (matplotlib, scipy):

```bash
pip install egora[diagnostics]
```

For development:

```bash
pip install egora[dev]
```

## Quick Start

### Training with EgoRA

```python
from transformers import AutoModelForCausalLM
from egora import apply_lora, get_total_egora_penalty, refresh_all_shadows
from egora import EntropyGovernor

# Load model and apply EgoRA-LoRA adapters
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
lora_modules = apply_lora(model, rank=16, use_egora=True)
gov = EntropyGovernor(alpha=1.359, lam_floor=0.6931)

# Training loop
for batch in dataloader:
    outputs = model(**batch)
    task_loss = outputs.loss

    # Dynamic entropy-governed penalty
    lam = gov.compute_lambda(outputs.logits)
    penalty = get_total_egora_penalty(lora_modules)
    total_loss = task_loss + lam * penalty

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Periodically refresh shadow matrices
refresh_all_shadows(lora_modules, momentum=0.9)
```

### Post-Training Diagnostics

```python
from egora import compute_head_geometry

results = compute_head_geometry(base_model, tuned_model)
print(f"Mean rotation: {results['theta_bar_deg']:.2f} deg")
print(f"Damaged heads: {results['damaged_fraction']*100:.1f}%")
```

## Adapter Types

| Adapter | Class | Description |
|---------|-------|-------------|
| **EgoRA-LoRA** | `EgoRALoRALinear` | Standard LoRA + entropy-governed orthogonality penalty |
| **DoRA** | `DoRALinear` | Weight-Decomposed LoRA (Liu et al., 2024) |
| **rsLoRA** | `rsLoRALinear` | Rank-Stabilized LoRA (Kalajdzievski, 2023) |

## Command-Line Interface

EgoRA includes a CLI for quick diagnostics without writing Python code:

```bash
# Compare base vs fine-tuned model
egora diagnose meta-llama/Llama-3.2-1B ./my-finetuned-model -o results.json --plot

# Show model architecture info (layers, d_head, θ_crit, LoRA param estimates)
egora info meta-llama/Llama-3.2-1B

# Version
egora version
```

## Visualization

Generate publication-quality rotation geometry plots (requires `pip install egora[diagnostics]`):

```python
from egora.plotting import plot_rotation_report, plot_rotation_heatmap

# Combined 4-panel report: heatmap, layer profile, modes, projections
fig = plot_rotation_report(results)
fig.savefig("rotation_report.png", dpi=150)

# Individual plots
fig = plot_rotation_heatmap(results)     # layer × projection heatmap
fig = plot_layer_profile(results)        # rotation across depth
fig = plot_mode_distribution(results)    # preserved/additive/substitutive/damaged
fig = plot_projection_comparison(results) # Q vs K vs V vs O
```

## PEFT-Compatible API

For users familiar with HuggingFace PEFT, EgoRA provides a compatible wrapper:

```python
from egora import EgoRAPeftModel, EgoRALoraConfig

config = EgoRALoraConfig(r=16, lora_alpha=32, use_egora=True)
model = EgoRAPeftModel.from_pretrained("meta-llama/Llama-3.2-1B", config)

model.print_trainable_parameters()
# trainable params: 6,553,600 || all params: 1,235,814,400 || trainable%: 0.53

# One-liner training loss
total_loss = model.compute_total_loss(task_loss, logits)

# Save / load adapters
model.save_pretrained("./my-egora-adapter")
model.load_adapter("./my-egora-adapter")

# Merge into base weights for deployment
base_model = model.merge_and_unload()
```

## Key Concepts

- **Entropy Governor**: Dynamically scales the regularization penalty based on model uncertainty — high entropy (uncertain) → strong penalty, low entropy (confident) → relaxed penalty.
- **Shadow Matrix**: A pseudo-inverse reference that tracks the adapter's structural orientation, enabling measurement of spectral conditioning.
- **Rotation-Retention Law**: $\Delta M \propto \bar{\theta}$ — the empirical law linking representational rotation to benchmark performance change. Critical threshold: $\theta_{\text{crit}} \approx 5°$.
- **Learning Modes**: Per-head classification into additive, substitutive, preserved, or damaged based on rotation angle and magnitude ratio.

## API Reference

### Core Functions

- `apply_lora(model, rank, use_egora, ...)` — Replace attention projections with LoRA adapters
- `get_total_egora_penalty(lora_modules)` — Sum penalty across all adapter modules
- `refresh_all_shadows(lora_modules, momentum)` — Update shadow matrices
- `merge_lora(model, lora_modules)` / `unmerge_lora(replacements)` — Merge/restore adapters

### Governor

- `EntropyGovernor(alpha, lam_floor, ...)` — Dynamic scaling controller
- `GovernorConfig(...)` — Configuration dataclass with adaptive alpha support

### Diagnostics

- `compute_head_geometry(base_model, tuned_model)` — Per-head rotation analysis
- `compute_interhead_diversity(model)` — Head diversity matrix
- `save_geometry(results, output_path)` — Export results to JSON + numpy

## Examples

See the [`examples/`](https://github.com/ArsSocratica/EgoRA/tree/main/examples) directory:

- **`quickstart_training.py`** — End-to-end fine-tuning with EgoRA + post-training diagnostics
- **`diagnostics_only.py`** — Standalone rotation analysis between any two checkpoints
- **`peft_compat_example.py`** — PEFT-style API demo with save/load/merge

## Resources

| | Link |
|---|---|
| 📦 **PyPI** | [`pip install egora`](https://pypi.org/project/egora/) |
| 💻 **GitHub** | [ArsSocratica/EgoRA](https://github.com/ArsSocratica/EgoRA) |
| 🤗 **Dataset** | [ArsSocratica/egora-benchmarks](https://huggingface.co/datasets/ArsSocratica/egora-benchmarks) — benchmark results, rotation geometry, training curves |
| 📄 **Paper** | [arXiv:2602.05192](https://arxiv.org/abs/2602.05192) |

### Benchmark Dataset

The full experiment data is published on HuggingFace:

- **Llama 3.2 1B / 3B, Llama 3.1 8B** — Alpaca + Medical domain, 4 methods (Baseline LoRA, DoRA, EgoRA e², EgoRA adaptive v2)
- **Cross-modal** — Mistral-7B, Phi-3 Mini, LLaVA-7B, Qwen2-VL-7B
- **Rotation geometry** — per-head θ, learning modes, knowledge maps, alignment landscapes
- **Threshold analysis** — Rotation-Retention Law validation, dimensionality threshold, phase transition

```python
# Load with HuggingFace datasets
from datasets import load_dataset
ds = load_dataset("ArsSocratica/egora-benchmarks")
```

## Citation

If you use EgoRA in your research, please cite:

```bibtex
@article{dillerop2026rotation,
  title={The Rotation-Retention Law: Knowledge Loss Is Proportional to 
         Representational Rotation in Fine-Tuned Language Models.
         With EgoRA: Entropy-Governed Orthogonality Regularization 
         for Adaptation},
  author={Dillerop, Mark},
  year={2026},
  note={Preprint}
}
```

## License

This software is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**, with an **Additional Permission for Academic Use** pursuant to AGPL Section 7.

- **Academic use**: Free, no copyleft obligations, citation required. See [LICENSE-ACADEMIC](https://github.com/ArsSocratica/EgoRA/blob/main/LICENSE-ACADEMIC).
- **Commercial use**: Requires both a software license and a patent license.

### Patent Notice

The methods implemented in this software are covered by U.S. Provisional Patent Application No. 64/024,742 ("Entropy-Governed Orthogonality Regularization for Knowledge-Preserving Neural Network Adaptation and Rotation-Retention Diagnostic Framework"), filed April 1, 2026, by Mark Dillerop.

Academic use is permitted without a separate patent license. Commercial use requires both a software license and a patent license.

**Contact**: mark@dillerop.com
