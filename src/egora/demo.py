"""
EgoRA Gradio Dashboard
======================
Interactive rotation diagnostics, model info, and a live EgoRA vs plain LoRA
comparison. Launch with ``egora demo`` or::

    from egora.demo import build_demo, launch_demo
    launch_demo(build_demo())

Requires ``gradio`` (and ``transformers``; ``peft`` for PEFT adapters).
"""

import copy
import functools
import json
import os
import tempfile
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch

from egora._version import __version__

EXAMPLE_PAIRS = [
    ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"],
    ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct"],
]
DEFAULT_COMPARE_MODEL = "HuggingFaceTB/SmolLM2-135M"
BUILTIN_DATASET = "builtin:medical"

# Narrow-domain corpus for the comparison tab (fine-tuning target).
DOMAIN_TEXTS = [
    "Hypertension is diagnosed when blood pressure is persistently at or above 130/80 mmHg.",
    "Metformin is a first-line oral medication for type 2 diabetes mellitus.",
    "Atrial fibrillation increases the risk of stroke and often requires anticoagulation.",
    "Beta blockers reduce heart rate and myocardial oxygen demand in angina.",
    "Community-acquired pneumonia is commonly caused by Streptococcus pneumoniae.",
    "An elevated troponin level indicates injury to the heart muscle.",
    "Asthma is treated with inhaled corticosteroids and short-acting bronchodilators.",
    "Chronic kidney disease is staged by the estimated glomerular filtration rate.",
    "Iron deficiency anemia presents with fatigue, pallor, and low ferritin levels.",
    "Statins lower LDL cholesterol by inhibiting HMG-CoA reductase.",
    "Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response.",
    "Hypothyroidism is treated with daily oral levothyroxine replacement.",
    "Warfarin therapy is monitored using the international normalized ratio.",
    "Deep vein thrombosis can lead to pulmonary embolism if the clot travels to the lungs.",
    "ACE inhibitors may cause a dry cough by increasing bradykinin levels.",
    "Gout is caused by deposition of monosodium urate crystals in the joints.",
    "Appendicitis classically begins with periumbilical pain that migrates to the right.",
    "Insulin is required for all patients with type 1 diabetes mellitus.",
    "Chronic obstructive pulmonary disease is strongly associated with smoking.",
    "Loop diuretics such as furosemide act on the thick ascending limb of Henle.",
    "Migraine headaches are often unilateral, pulsating, and accompanied by nausea.",
    "Osteoporosis increases fracture risk and is assessed with a DEXA scan.",
    "Proton pump inhibitors reduce gastric acid secretion in peptic ulcer disease.",
    "Rheumatoid arthritis is a symmetric inflammatory polyarthritis of small joints.",
]

# General-knowledge probe: loss increase here is a proxy for forgetting.
PROBE_TEXTS = [
    "The capital of France is Paris, a city known for the Eiffel Tower.",
    "Water boils at 100 degrees Celsius at sea level.",
    "William Shakespeare wrote Romeo and Juliet and Hamlet.",
    "The Earth orbits the Sun once every 365 days.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Python is a popular programming language created by Guido van Rossum.",
    "The Great Wall of China stretches thousands of kilometers.",
    "Mount Everest is the highest mountain above sea level.",
    "The French Revolution began in 1789.",
]

_MODE_ORDER = ["preserved", "additive", "substitutive", "damaged"]
_VERDICT_ICON = {"safe": "✅", "warning": "⚠️", "danger": "❌"}


def pick_device():
    """Best available device and model dtype for training."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


class _ModelCache:
    """Small LRU cache of loaded base models (they are expensive to load)."""

    def __init__(self, max_items: int = 2):
        self._items: "OrderedDict[tuple, torch.nn.Module]" = OrderedDict()
        self._max = max_items
        self._lock = threading.Lock()

    def get(self, name: str, trust_remote_code: bool, dtype: Any = "auto", device: str = "cpu"):
        from egora.loading import load_causal_lm

        key = (name, trust_remote_code, str(dtype), device)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            model = load_causal_lm(name, dtype, device, trust_remote_code)
            self._items[key] = model
            while len(self._items) > self._max:
                self._items.popitem(last=False)
            return model


_CACHE = _ModelCache()


# ── Diagnose ─────────────────────────────────────────────────────────────────


def format_report_markdown(results: Dict[str, Any], tuned_kind: str = "model") -> str:
    unit = "Heads" if results.get("granularity") == "head" else "Projections"
    lines = [
        "## Rotation Geometry Report",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| {unit} analyzed | {results['n_heads']} |",
        f"| Mean rotation (θ̄) | {results['theta_bar_deg']:.2f}° |",
        f"| Std rotation | {results['theta_std_deg']:.2f}° |",
        f"| Min / Max | {results['theta_min_deg']:.2f}° / {results['theta_max_deg']:.2f}° |",
    ]
    if results.get("theta_crit_deg"):
        lines.append(
            f"| d_head / model θ_crit | {results['head_dim']} / {results['theta_crit_deg']:.2f}° |"
        )
    lines.append(f"| Damage threshold | {results['damage_threshold_deg']:.2f}° |")
    if tuned_kind != "model":
        lines.append(f"| Fine-tuned input | {tuned_kind.replace('_', ' ')} (merged) |")
    lines += ["", "### Learning Modes"]
    for mode in _MODE_ORDER:
        count = results["mode_counts"].get(mode, 0)
        frac = results["mode_fractions"].get(mode, 0.0)
        lines.append(f"- **{mode}**: {count} ({frac * 100:.1f}%) {'█' * int(frac * 20)}")
    v = results["verdict"]
    lines += ["", "### Verdict", f"{_VERDICT_ICON[v['level']]} {v['message']}"]
    return "\n".join(lines)


def layer_table(results: Dict[str, Any]):
    import pandas as pd

    rows: Dict[int, Dict[str, List[float]]] = {}
    for h in results["heads"]:
        if h["layer"] is None:
            continue
        rows.setdefault(h["layer"], {}).setdefault(h["proj_type"], []).append(
            h["rotation_angle_deg"]
        )
    projs = sorted({h["proj_type"] for h in results["heads"]})
    data = []
    for layer in sorted(rows):
        all_angles = [a for v in rows[layer].values() for a in v]
        row = {"layer": layer, "mean θ (°)": round(sum(all_angles) / len(all_angles), 3)}
        for p in projs:
            vals = rows[layer].get(p, [])
            row[f"{p} θ (°)"] = round(sum(vals) / len(vals), 3) if vals else None
        data.append(row)
    return pd.DataFrame(data)


def top_units_table(results: Dict[str, Any], n: int = 25):
    import pandas as pd

    heads = sorted(results["heads"], key=lambda h: h["rotation_angle_deg"], reverse=True)[:n]
    return pd.DataFrame([
        {
            "layer": h["layer"],
            "projection": h["proj_type"],
            "head": "" if h.get("head") is None else h["head"],
            "θ (°)": round(h["rotation_angle_deg"], 3),
            "magnitude ratio": round(h["magnitude_ratio"], 4),
            "mode": h["learning_mode"],
        }
        for h in heads
    ])


def run_diagnostics(
    base_name: str,
    tuned_ref: str,
    granularity: str = "projection",
    use_model_crit: bool = True,
    damage_threshold: float = 5.0,
    trust_remote_code: bool = False,
):
    """Generator yielding ``(summary_md, figure, layer_df, units_df, files)``."""
    from egora.geometry import compute_head_geometry
    from egora.loading import load_tuned_model
    from egora.model_info import get_attention_dims, theta_crit_deg

    empty = (None, None, None, None)
    base_name, tuned_ref = (base_name or "").strip(), (tuned_ref or "").strip()
    if not base_name or not tuned_ref:
        yield ("Please provide both a base model and a fine-tuned model or adapter.",) + empty
        return
    try:
        yield ("Loading base model...",) + empty
        base = _CACHE.get(base_name, trust_remote_code)

        yield ("Loading fine-tuned model / adapter...",) + empty
        tuned, kind = load_tuned_model(base, tuned_ref, trust_remote_code=trust_remote_code)

        head_dim = get_attention_dims(base)["head_dim"]
        model_crit = theta_crit_deg(head_dim) if head_dim else None
        verdict_crit = model_crit if (use_model_crit and model_crit) else 5.0
        threshold = verdict_crit if use_model_crit else float(damage_threshold)

        yield (f"Computing {granularity}-level rotation geometry...",) + empty
        results = compute_head_geometry(
            base, tuned,
            damage_threshold_deg=threshold,
            granularity=granularity,
            head_dim=head_dim,
            verdict_theta_crit=verdict_crit,
        )
        del tuned
        if "error" in results:
            yield (f"Error: {results['error']}",) + empty
            return

        fig = None
        files = []
        out_dir = tempfile.mkdtemp(prefix="egora-")
        json_path = os.path.join(out_dir, "egora_geometry.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        files.append(json_path)
        try:
            from egora.plotting import plot_rotation_report

            fig = plot_rotation_report(results)
            png_path = os.path.join(out_dir, "egora_rotation_report.png")
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            files.append(png_path)
        except ImportError:
            pass

        yield (
            format_report_markdown(results, kind),
            fig,
            layer_table(results),
            top_units_table(results),
            files,
        )
    except Exception as e:  # surfaced in the UI
        yield (f"Error: {e}",) + empty


# ── Model info ───────────────────────────────────────────────────────────────


def run_model_info(model_name: str, trust_remote_code: bool = False) -> str:
    from egora.model_info import format_model_info_markdown, load_model_skeleton, summarize_model

    model_name = (model_name or "").strip()
    if not model_name:
        return "Please provide a model name."
    try:
        model = load_model_skeleton(model_name, trust_remote_code=trust_remote_code)
        return format_model_info_markdown(model_name, summarize_model(model))
    except Exception as e:
        return f"Error: {e}"


# ── EgoRA vs plain LoRA ──────────────────────────────────────────────────────


def _load_texts(dataset: str, text_column: str, n_samples: int) -> List[str]:
    dataset = (dataset or "").strip()
    if not dataset or dataset == BUILTIN_DATASET:
        return list(DOMAIN_TEXTS)
    from datasets import load_dataset

    ds = load_dataset(dataset, split=f"train[:{int(n_samples)}]")
    col = text_column if text_column in ds.column_names else ds.column_names[0]
    return [str(t) for t in ds[col] if t]


def _comparison_figure(history: Dict[str, Dict[str, list]], theta_crit: Optional[float]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"LoRA": "#757575", "EgoRA": "#1565C0"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("theta", "Mean rotation θ̄ (°)", "Representational rotation"),
        ("loss", "Train loss", "Task loss (domain data)"),
        ("probe_delta", "Δ loss vs base", "Forgetting (general-text probe)"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        for method, h in history.items():
            if h["step"]:
                ax.plot(h["step"], h[key], marker=".", label=method, color=colors.get(method))
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3)
    if theta_crit:
        axes[0].axhline(theta_crit, color="red", linestyle="--", linewidth=1,
                        label=f"θ_crit = {theta_crit:.2f}°")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[0].legend()
    fig.tight_layout()
    return fig


def _comparison_table(history):
    import pandas as pd

    rows = []
    for method, h in history.items():
        if not h["step"]:
            continue
        rows.append({
            "method": method,
            "steps": h["step"][-1],
            "final θ̄ (°)": round(h["theta"][-1], 3),
            "max θ (°)": round(h["theta_max"][-1], 3),
            "final train loss": round(h["loss"][-1], 4),
            "probe Δ loss": round(h["probe_delta"][-1], 4),
        })
    return pd.DataFrame(rows)


def run_comparison(
    model_name: str = DEFAULT_COMPARE_MODEL,
    dataset: str = BUILTIN_DATASET,
    text_column: str = "text",
    n_samples: int = 200,
    steps: int = 40,
    rank: int = 8,
    lr: float = 5e-4,
    max_length: int = 64,
    batch_size: int = 4,
    shadow_refresh: int = 10,
    seed: int = 0,
    eval_every: int = 5,
    trust_remote_code: bool = False,
):
    """Train plain LoRA and EgoRA-LoRA with identical seeds and data.

    Generator yielding ``(status_md, figure, table_df)`` as training progresses.
    """
    from transformers import AutoTokenizer

    from egora.adapters import apply_lora, get_total_egora_penalty, refresh_all_shadows
    from egora.geometry import compute_adapter_rotation
    from egora.governor import EntropyGovernor
    from egora.model_info import get_attention_dims, theta_crit_deg

    steps, rank, batch_size = int(steps), int(rank), int(batch_size)
    max_length, shadow_refresh = int(max_length), max(1, int(shadow_refresh))
    eval_every, seed = max(1, int(eval_every)), int(seed)

    history = {
        m: {"step": [], "theta": [], "theta_max": [], "loss": [], "probe_delta": []}
        for m in ("LoRA", "EgoRA")
    }
    try:
        yield "Loading tokenizer, data and model...", None, None
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        texts = _load_texts(dataset, text_column, n_samples)

        def encode(batch_texts):
            enc = tok(batch_texts, truncation=True, max_length=max_length,
                      padding="max_length", return_tensors="pt")
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            return enc["input_ids"], enc["attention_mask"], labels

        device, dtype = pick_device()
        base = _CACHE.get(model_name, trust_remote_code, dtype=dtype, device=device)
        head_dim = get_attention_dims(base)["head_dim"]
        crit = theta_crit_deg(head_dim) if head_dim else None

        ids, mask, labels = (t.to(device) for t in encode(texts))
        p_ids, p_mask, p_labels = (t.to(device) for t in encode(PROBE_TEXTS))

        def probe_loss(model):
            model.eval()
            with torch.no_grad():
                return model(input_ids=p_ids, attention_mask=p_mask, labels=p_labels).loss.item()

        base_probe = probe_loss(base)

        for method, use_egora in (("LoRA", False), ("EgoRA", True)):
            torch.manual_seed(seed)
            model = copy.deepcopy(base)
            modules = apply_lora(model, rank=rank, lora_alpha=2 * rank,
                                 use_egora=use_egora, lora_dropout=0.0)
            gov = EntropyGovernor()
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=float(lr))
            gen = torch.Generator().manual_seed(seed)
            h = history[method]

            for step in range(1, steps + 1):
                model.train()
                idx = torch.randint(0, ids.shape[0], (batch_size,), generator=gen).to(device)
                out = model(input_ids=ids[idx], attention_mask=mask[idx], labels=labels[idx])
                total = out.loss
                if use_egora:
                    lam = gov.compute_lambda(out.logits)
                    total = total + lam * get_total_egora_penalty(modules)
                total.backward()
                opt.step()
                opt.zero_grad()
                if use_egora and step % shadow_refresh == 0:
                    refresh_all_shadows(modules, momentum=0.9)

                if step % eval_every == 0 or step == steps:
                    rot = compute_adapter_rotation(modules)
                    h["step"].append(step)
                    h["theta"].append(rot["theta_bar_deg"])
                    h["theta_max"].append(rot["theta_max_deg"])
                    h["loss"].append(out.loss.item())
                    h["probe_delta"].append(probe_loss(model) - base_probe)
                    status = (
                        f"**{method}** step {step}/{steps} on {device.upper()} — "
                        f"θ̄ = {rot['theta_bar_deg']:.3f}°, loss = {out.loss.item():.3f}"
                    )
                    yield status, _comparison_figure(history, crit), _comparison_table(history)
            del model, opt

        yield (
            f"Done. Both runs used the same seed, data order and initialization "
            f"({len(texts)} training texts, {steps} steps, rank {rank}, lr {lr}).",
            _comparison_figure(history, crit),
            _comparison_table(history),
        )
    except Exception as e:
        yield f"Error: {e}", None, None


# ── UI ───────────────────────────────────────────────────────────────────────


def _gradio_major() -> int:
    import gradio as gr

    try:
        return int(gr.__version__.split(".")[0])
    except ValueError:
        return 0


def build_demo(trust_remote_code: bool = False, allow_training: bool = True):
    """Build the Gradio Blocks app."""
    import gradio as gr

    theme = gr.themes.Soft()
    blocks_kwargs = {"title": f"EgoRA v{__version__} — Rotation Diagnostics"}
    if _gradio_major() < 6:
        blocks_kwargs["theme"] = theme

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(f"""
# EgoRA — Rotation Diagnostics Dashboard
**v{__version__}** | [GitHub](https://github.com/ArsSocratica/EgoRA) | \
[PyPI](https://pypi.org/project/egora/) | [DOI](https://doi.org/10.5281/zenodo.19410504)

Fine-tune without forgetting. Measure how much a model's internal representations
rotated during training.
        """)

        with gr.Tabs():
            with gr.Tab("Diagnose"):
                gr.Markdown(
                    "Compare a base model with a fine-tuned version. The fine-tuned input can "
                    "be a full model, a PEFT LoRA adapter, or an EgoRA adapter folder "
                    "(Hub id or path)."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        base_in = gr.Textbox(label="Base model", value=EXAMPLE_PAIRS[0][0])
                        tuned_in = gr.Textbox(
                            label="Fine-tuned model or adapter",
                            placeholder="org/model, org/lora-adapter, or ./path",
                        )
                        gr.Examples(examples=EXAMPLE_PAIRS, inputs=[base_in, tuned_in])
                        gran_in = gr.Radio(
                            ["projection", "head"], value="projection", label="Granularity",
                            info="projection = whole q/k/v/o matrices; head = per attention head",
                        )
                        crit_in = gr.Checkbox(
                            value=True, label="Use model θ_crit = arcsin(1/√d_head)",
                            info="Unchecked: fixed 5°/3° verdict and the slider below",
                        )
                        thr_in = gr.Slider(1.0, 15.0, value=5.0, step=0.5,
                                           label="Damage threshold (°)", interactive=True)
                        diag_btn = gr.Button("Run diagnostics", variant="primary")
                    with gr.Column(scale=2):
                        summary_out = gr.Markdown()
                        plot_out = gr.Plot(label="Rotation report")
                        files_out = gr.File(label="Downloads (JSON, PNG)", file_count="multiple")
                with gr.Accordion("Per-layer rotation", open=False):
                    layers_out = gr.Dataframe(interactive=False)
                with gr.Accordion("Most rotated units", open=False):
                    units_out = gr.Dataframe(interactive=False)

                diag_btn.click(
                    fn=functools.partial(run_diagnostics, trust_remote_code=trust_remote_code),
                    inputs=[base_in, tuned_in, gran_in, crit_in, thr_in],
                    outputs=[summary_out, plot_out, layers_out, units_out, files_out],
                )

            with gr.Tab("Model Info"):
                gr.Markdown(
                    "Architecture facts relevant to EgoRA. Reads only the config: no weights are "
                    "downloaded."
                )
                info_in = gr.Textbox(label="Model name", value=EXAMPLE_PAIRS[0][0])
                info_btn = gr.Button("Get info", variant="primary")
                info_out = gr.Markdown()
                info_btn.click(
                    fn=lambda m: run_model_info(m, trust_remote_code),
                    inputs=[info_in], outputs=[info_out],
                )

            if allow_training:
                with gr.Tab("EgoRA vs LoRA"):
                    gr.Markdown(
                        "Fine-tune the same small model twice — plain LoRA and EgoRA-LoRA — with "
                        "identical seed, initialization and data order. Tracks mean rotation θ̄ "
                        "of the adapted projections, the task loss, and the loss change on a "
                        "general-text probe as a forgetting proxy (a rough signal at this scale). "
                        "The default (SmolLM2-135M, 40 steps) runs in about a minute on CPU."
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            cmp_model = gr.Textbox(label="Model", value=DEFAULT_COMPARE_MODEL)
                            cmp_data = gr.Textbox(
                                label="Dataset",
                                value=BUILTIN_DATASET,
                                info=f"'{BUILTIN_DATASET}' (24 built-in sentences) or a HF "
                                     "dataset id",
                            )
                            cmp_col = gr.Textbox(label="Text column", value="text")
                            cmp_n = gr.Slider(16, 2000, value=200, step=8, label="HF samples")
                            cmp_steps = gr.Slider(10, 300, value=40, step=5, label="Steps")
                            cmp_rank = gr.Slider(2, 64, value=8, step=2, label="LoRA rank")
                            cmp_lr = gr.Number(value=5e-4, label="Learning rate")
                            cmp_len = gr.Slider(16, 256, value=64, step=16, label="Max length")
                            cmp_bs = gr.Slider(1, 16, value=4, step=1, label="Batch size")
                            cmp_refresh = gr.Slider(
                                1, 100, value=10, step=1,
                                label="EgoRA shadow refresh (steps)",
                                info="The penalty is inactive until the first refresh "
                                     "(library default: 100)",
                            )
                            cmp_seed = gr.Number(value=0, precision=0, label="Seed")
                            cmp_btn = gr.Button("Run comparison", variant="primary")
                        with gr.Column(scale=2):
                            cmp_status = gr.Markdown()
                            cmp_plot = gr.Plot(label="EgoRA vs LoRA")
                            cmp_table = gr.Dataframe(interactive=False)

                    cmp_btn.click(
                        fn=functools.partial(
                            run_comparison, trust_remote_code=trust_remote_code
                        ),
                        inputs=[cmp_model, cmp_data, cmp_col, cmp_n, cmp_steps, cmp_rank,
                                cmp_lr, cmp_len, cmp_bs, cmp_refresh, cmp_seed],
                        outputs=[cmp_status, cmp_plot, cmp_table],
                    )

            with gr.Tab("About"):
                gr.Markdown("""
## About EgoRA

**EgoRA** (Entropy-Governed Orthogonality Regularization for Adaptation) is a dynamic
regularization method for fine-tuning neural networks.

The **Rotation-Retention Law**: knowledge loss in fine-tuned language models is
proportional to representational rotation (ΔM ∝ θ̄). The critical angle is
θ_crit = arcsin(1/√d_head) — about 5° for d_head = 128.

### Quick Start

```bash
pip install egora
egora train meta-llama/Llama-3.2-1B tatsu-lab/alpaca --epochs 1 --rank 16
```

### License
AGPL-3.0 with Academic Additional Permission. Free for research (citation required).

**Contact**: mark@dillerop.com
                """)

    demo._egora_theme = theme
    return demo


def launch_demo(demo, **launch_kwargs):
    """Launch the app, passing the theme where the installed Gradio expects it."""
    if _gradio_major() >= 6:
        launch_kwargs.setdefault("theme", getattr(demo, "_egora_theme", None))
    demo.queue().launch(**launch_kwargs)
