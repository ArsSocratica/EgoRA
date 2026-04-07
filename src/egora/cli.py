"""
EgoRA Command-Line Interface
=============================
Provides ``egora`` CLI commands for diagnostics and quick experimentation.

Commands:
  egora train MODEL DATASET                — Fine-tune a model with EgoRA
  egora diagnose BASE_MODEL TUNED_MODEL    — Compute rotation geometry
  egora info MODEL                         — Show model architecture summary
  egora demo                               — Launch Gradio web demo
  egora version                            — Print version
"""

import argparse
import json
import sys
import os
from typing import Optional


def _load_model(path_or_name: str):
    """Load a HuggingFace model from path or Hub name."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("Error: `transformers` is required for model loading.")
        print("Install with: pip install transformers")
        sys.exit(1)

    import torch

    print(f"Loading model: {path_or_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        path_or_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model


def cmd_diagnose(args):
    """Run rotation geometry diagnostics between base and tuned model."""
    from egora.geometry import compute_head_geometry, compute_interhead_diversity, save_geometry

    base_model = _load_model(args.base_model)
    tuned_model = _load_model(args.tuned_model)

    target_modules = args.target_modules.split(",") if args.target_modules else None

    print("Computing head geometry ...")
    results = compute_head_geometry(
        base_model, tuned_model,
        target_modules=target_modules,
        damage_threshold_deg=args.damage_threshold,
    )

    if "error" in results:
        print(f"Error: {results['error']}")
        sys.exit(1)

    # Print summary
    print()
    print("=" * 60)
    print("  EgoRA Rotation Geometry Report")
    print("=" * 60)
    print(f"  Heads analyzed:       {results['n_heads']}")
    print(f"  Mean rotation (θ̄):    {results['theta_bar_deg']:.2f}°")
    print(f"  Std rotation:         {results['theta_std_deg']:.2f}°")
    print(f"  Min / Max:            {results['theta_min_deg']:.2f}° / {results['theta_max_deg']:.2f}°")
    print()
    print("  Learning Modes:")
    for mode, count in results['mode_counts'].items():
        frac = results['mode_fractions'][mode]
        bar = "█" * int(frac * 30)
        print(f"    {mode:14s}  {count:4d}  ({frac*100:5.1f}%)  {bar}")
    print()

    # Rotation-Retention Law prediction
    theta_bar = results['theta_bar_deg']
    if theta_bar < 3.0:
        verdict = "✅ SAFE — Knowledge well preserved (θ̄ < 3°)"
    elif theta_bar < 5.0:
        verdict = "⚠️  WARNING — Moderate rotation, check benchmarks (3° < θ̄ < 5°)"
    else:
        verdict = "❌ DANGER — Significant knowledge loss expected (θ̄ > 5°)"
    print(f"  Verdict: {verdict}")
    print()

    # Per-projection breakdown
    if results.get('per_projection_mean_rotation'):
        print("  Per-Projection Mean Rotation:")
        for proj, angle in results['per_projection_mean_rotation'].items():
            print(f"    {proj:10s}  {angle:.2f}°")
        print()

    # Per-layer summary (first/last 3)
    layer_means = results.get('per_layer_mean_rotation', {})
    if layer_means:
        layers = sorted(layer_means.items(), key=lambda x: int(x[0]))
        n = len(layers)
        print(f"  Per-Layer Mean Rotation ({n} layers):")
        show = layers[:3] + ([("...", None)] if n > 6 else []) + layers[-3:] if n > 6 else layers
        for layer, angle in show:
            if angle is None:
                print(f"    {'...'}")
            else:
                print(f"    Layer {int(layer):3d}:  {angle:.2f}°")
        print()

    # Interhead diversity
    if args.diversity:
        print("Computing interhead diversity ...")
        sim_matrix, head_names = compute_interhead_diversity(
            tuned_model, target_modules=target_modules
        )
        if sim_matrix is not None:
            import numpy as np
            off_diag = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
            print(f"  Mean off-diagonal similarity: {off_diag.mean():.4f}")
            print(f"  Diversity score: {1 - off_diag.mean():.4f}")
            results['interhead_diversity'] = sim_matrix.tolist()
        print()

    # Save results
    if args.output:
        save_geometry(results, args.output)
        print(f"Results saved to: {args.output}")

    # Plot if requested
    if args.plot:
        try:
            from egora.plotting import plot_rotation_report
            fig = plot_rotation_report(results)
            plot_path = args.output.replace('.json', '.png') if args.output else 'egora_report.png'
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
        except ImportError:
            print("Warning: matplotlib required for plotting. Install with: pip install egora[diagnostics]")

    print("=" * 60)


def cmd_info(args):
    """Show model architecture summary relevant to EgoRA."""
    model = _load_model(args.model)

    target_modules = args.target_modules.split(",") if args.target_modules else \
        ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    total_params = sum(p.numel() for p in model.parameters())
    target_params = 0
    target_count = 0

    print()
    print("=" * 60)
    print("  EgoRA Model Info")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Total params:   {total_params:,}")
    print()

    layers_found = set()
    for name, param in model.named_parameters():
        for tm in target_modules:
            if (f".{tm}." in name or name.endswith(f".{tm}.weight")) and name.endswith('.weight'):
                target_params += param.numel()
                target_count += 1
                for part in name.split('.'):
                    if part.isdigit():
                        layers_found.add(int(part))
                        break

    d_head = None
    for name, param in model.named_parameters():
        if 'q_proj.weight' in name:
            d_head = param.shape[0]  # out_features
            break

    print(f"  Target modules: {', '.join(target_modules)}")
    print(f"  Target layers:  {len(layers_found)} ({min(layers_found) if layers_found else '?'}-{max(layers_found) if layers_found else '?'})")
    print(f"  Target params:  {target_params:,} ({target_params/total_params*100:.1f}%)")
    print(f"  Projections:    {target_count}")
    if d_head:
        import math
        theta_crit = math.degrees(math.asin(1.0 / math.sqrt(d_head)))
        print(f"  d_head:         {d_head}")
        print(f"  θ_crit:         {theta_crit:.2f}° (arcsin(1/√d_head))")
    print()

    # LoRA parameter estimate
    for rank in [8, 16, 32, 64]:
        lora_params = target_count * (d_head * rank + rank * d_head) if d_head else 0
        pct = lora_params / total_params * 100 if total_params else 0
        print(f"  LoRA rank={rank:2d}: {lora_params:>12,} trainable params ({pct:.2f}%)")

    print()
    print("=" * 60)


def cmd_train(args):
    """Fine-tune a model with EgoRA in one command."""
    try:
        from transformers import AutoTokenizer
        from datasets import load_dataset
    except ImportError:
        print("Error: `transformers` and `datasets` are required for training.")
        print("Install with: pip install transformers datasets accelerate")
        sys.exit(1)

    import torch
    from egora.trainer import EgoRATrainer, EgoRATrainingArguments
    from egora.peft_compat import EgoRALoraConfig

    # Load tokenizer
    print(f"Loading tokenizer: {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"Loading dataset: {args.dataset} ...")
    split = f"train[:{args.max_samples}]" if args.max_samples else "train"
    try:
        ds = load_dataset(args.dataset, split=split)
    except Exception:
        ds = load_dataset(args.dataset, split=split, trust_remote_code=True)

    # Tokenize
    text_col = args.text_column
    if text_col not in ds.column_names:
        candidates = ["text", "instruction", "content", "prompt", "input"]
        for c in candidates:
            if c in ds.column_names:
                text_col = c
                break
        else:
            text_col = ds.column_names[0]
    print(f"Using text column: '{text_col}'")

    def tokenize_fn(example):
        tokens = tokenizer(
            example[text_col],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
    ds.set_format("torch")
    print(f"Dataset ready: {len(ds)} examples, max_length={args.max_length}")

    # Config
    egora_config = EgoRALoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        use_egora=not args.no_egora,
        adapter_type=args.adapter_type,
        lora_dropout=args.lora_dropout,
        shadow_refresh_interval=args.shadow_refresh,
        egora_alpha=args.egora_alpha,
        egora_lam_floor=args.egora_lam_floor,
        egora_adaptive_alpha=args.adaptive_alpha,
        egora_alpha_min=args.alpha_min,
        egora_alpha_max=args.alpha_max,
        egora_alpha_gamma=args.alpha_gamma,
        egora_alpha_cooldown=args.alpha_cooldown,
    )

    train_args = EgoRATrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_strategy="epoch" if args.save else "no",
        bf16=torch.cuda.is_available() and not args.fp32,
        gradient_accumulation_steps=args.gradient_accumulation,
    )

    model_kwargs = {}
    if torch.cuda.is_available() and not args.fp32:
        model_kwargs["torch_dtype"] = torch.bfloat16

    # Train
    trainer = EgoRATrainer(
        model_name=args.model,
        egora_config=egora_config,
        args=train_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        model_kwargs=model_kwargs,
    )
    trainer.print_trainable_parameters()

    print()
    print("=" * 60)
    print("  Starting EgoRA fine-tuning")
    print("=" * 60)
    trainer.train()

    # Diagnostics
    if not args.skip_diagnose:
        print()
        print("Running post-training diagnostics ...")
        try:
            report = trainer.diagnose()
            print(f"  Mean rotation:   {report['theta_bar_deg']:.2f}°")
            print(f"  Damaged heads:   {report['damaged_fraction']*100:.1f}%")
            theta_bar = report['theta_bar_deg']
            if theta_bar < 3.0:
                print("  Verdict: ✅ SAFE — Knowledge well preserved")
            elif theta_bar < 5.0:
                print("  Verdict: ⚠️  WARNING — Moderate rotation")
            else:
                print("  Verdict: ❌ DANGER — Significant knowledge loss expected")
        except Exception as e:
            print(f"  Diagnostics failed: {e}")

    # Save
    if args.save:
        trainer.save_model(args.output_dir)
        print(f"\nModel saved to: {args.output_dir}")

    # Merge
    if args.merge:
        print("Merging adapters into base model ...")
        merged = trainer.merge_and_unload()
        merge_path = os.path.join(args.output_dir, "merged")
        merged.save_pretrained(merge_path)
        tokenizer.save_pretrained(merge_path)
        print(f"Merged model saved to: {merge_path}")

    print("\n✅ Done!")


def cmd_demo(args):
    """Launch Gradio web demo for EgoRA diagnostics."""
    try:
        import gradio as gr
    except ImportError:
        print("Error: gradio is required for the demo.")
        print("Install with: pip install gradio")
        sys.exit(1)

    from egora._version import __version__

    def run_diagnostics(base_model_name, tuned_model_name, damage_threshold):
        """Run rotation diagnostics and return results."""
        if not base_model_name or not tuned_model_name:
            return "Please provide both model names.", None, None

        try:
            import torch
            from transformers import AutoModelForCausalLM
            from egora.geometry import compute_head_geometry

            yield "Loading base model...", None, None

            base = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.float32,
                device_map="cpu", trust_remote_code=True
            )

            yield "Loading tuned model...", None, None

            tuned = AutoModelForCausalLM.from_pretrained(
                tuned_model_name, torch_dtype=torch.float32,
                device_map="cpu", trust_remote_code=True
            )

            yield "Computing rotation geometry...", None, None

            results = compute_head_geometry(
                base, tuned, damage_threshold_deg=damage_threshold
            )

            # Format text summary
            theta_bar = results['theta_bar_deg']
            if theta_bar < 3.0:
                verdict = "✅ SAFE — Knowledge well preserved (θ̄ < 3°)"
            elif theta_bar < 5.0:
                verdict = "⚠️ WARNING — Moderate rotation (3° < θ̄ < 5°)"
            else:
                verdict = "❌ DANGER — Significant knowledge loss expected (θ̄ > 5°)"

            summary = f"""## Rotation Geometry Report

| Metric | Value |
|--------|-------|
| Heads analyzed | {results['n_heads']} |
| Mean rotation (θ̄) | {results['theta_bar_deg']:.2f}° |
| Std rotation | {results['theta_std_deg']:.2f}° |
| Min / Max | {results['theta_min_deg']:.2f}° / {results['theta_max_deg']:.2f}° |

### Learning Modes
"""
            for mode, count in results['mode_counts'].items():
                frac = results['mode_fractions'][mode]
                bar = "█" * int(frac * 20)
                summary += f"- **{mode}**: {count} ({frac*100:.1f}%) {bar}\n"

            summary += f"\n### Verdict\n{verdict}"

            # Generate plot
            plot_fig = None
            try:
                from egora.plotting import plot_rotation_report
                plot_fig = plot_rotation_report(results)
            except ImportError:
                pass

            yield summary, plot_fig, json.dumps(results, indent=2, default=str)

        except Exception as e:
            yield f"Error: {e}", None, None

    def run_model_info(model_name):
        """Get model info for EgoRA."""
        if not model_name:
            return "Please provide a model name."

        try:
            import torch
            import math
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32,
                device_map="cpu", trust_remote_code=True
            )
            total_params = sum(p.numel() for p in model.parameters())
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
            target_count = 0
            target_params = 0
            layers_found = set()

            for name, param in model.named_parameters():
                for tm in target_modules:
                    if (f".{tm}." in name or name.endswith(f".{tm}.weight")) and name.endswith('.weight'):
                        target_params += param.numel()
                        target_count += 1
                        for part in name.split('.'):
                            if part.isdigit():
                                layers_found.add(int(part))
                                break

            d_head = None
            for name, param in model.named_parameters():
                if 'q_proj.weight' in name:
                    d_head = param.shape[0]
                    break

            info = f"""## Model Info: {model_name}

| Property | Value |
|----------|-------|
| Total params | {total_params:,} |
| Target projections | {target_count} |
| Target params | {target_params:,} ({target_params/total_params*100:.1f}%) |
| Layers | {len(layers_found)} |
"""
            if d_head:
                theta_crit = math.degrees(math.asin(1.0 / math.sqrt(d_head)))
                info += f"| d_head | {d_head} |\n"
                info += f"| θ_crit | {theta_crit:.2f}° |\n"
                info += "\n### LoRA Parameter Estimates\n\n"
                info += "| Rank | Trainable Params | % of Total |\n"
                info += "|------|-----------------|------------|\n"
                for rank in [8, 16, 32, 64]:
                    lora_p = target_count * (d_head * rank + rank * d_head)
                    pct = lora_p / total_params * 100
                    info += f"| {rank} | {lora_p:,} | {pct:.2f}% |\n"

            del model
            return info

        except Exception as e:
            return f"Error: {e}"

    # Build Gradio interface
    with gr.Blocks(
        title=f"EgoRA v{__version__} — Rotation Diagnostics",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(f"""
# 🧠 EgoRA — Rotation Diagnostics Dashboard
**v{__version__}** | [GitHub](https://github.com/ArsSocratica/EgoRA) | [PyPI](https://pypi.org/project/egora/) | [DOI](https://doi.org/10.5281/zenodo.19410504)

Fine-tune without forgetting. Visualize how much your model's internal representations rotated during training.
        """)

        with gr.Tabs():
            with gr.Tab("🔍 Diagnose"):
                gr.Markdown("Compare a base model with a fine-tuned version to measure representational rotation.")
                with gr.Row():
                    with gr.Column():
                        base_input = gr.Textbox(
                            label="Base Model",
                            placeholder="meta-llama/Llama-3.2-1B",
                            value="meta-llama/Llama-3.2-1B",
                        )
                        tuned_input = gr.Textbox(
                            label="Fine-Tuned Model",
                            placeholder="path/to/your/finetuned-model",
                        )
                        threshold_slider = gr.Slider(
                            minimum=1.0, maximum=15.0, value=5.0, step=0.5,
                            label="Damage Threshold (degrees)",
                        )
                        diagnose_btn = gr.Button("🔬 Run Diagnostics", variant="primary")
                    with gr.Column():
                        summary_output = gr.Markdown(label="Summary")
                        plot_output = gr.Plot(label="Rotation Report")
                        json_output = gr.Code(label="Raw JSON", language="json")

                diagnose_btn.click(
                    fn=run_diagnostics,
                    inputs=[base_input, tuned_input, threshold_slider],
                    outputs=[summary_output, plot_output, json_output],
                )

            with gr.Tab("ℹ️ Model Info"):
                gr.Markdown("Get model architecture info relevant to EgoRA (layers, d_head, θ_crit, LoRA estimates).")
                info_input = gr.Textbox(
                    label="Model Name",
                    placeholder="meta-llama/Llama-3.2-1B",
                )
                info_btn = gr.Button("📊 Get Info", variant="primary")
                info_output = gr.Markdown(label="Model Info")

                info_btn.click(
                    fn=run_model_info,
                    inputs=[info_input],
                    outputs=[info_output],
                )

            with gr.Tab("📖 About"):
                gr.Markdown(f"""
## About EgoRA

**EgoRA** (Entropy-Governed Orthogonality Regularization for Adaptation) is a dynamic
regularization method for fine-tuning neural networks.

The **Rotation-Retention Law**: knowledge loss in fine-tuned language models is
proportional to representational rotation (ΔM ∝ θ̄).

### Quick Start

```bash
pip install egora
egora train meta-llama/Llama-3.2-1B tatsu-lab/alpaca --epochs 1 --rank 16
```

### Links
- **GitHub**: [ArsSocratica/EgoRA](https://github.com/ArsSocratica/EgoRA)
- **PyPI**: [egora](https://pypi.org/project/egora/)
- **DOI**: [10.5281/zenodo.19410504](https://doi.org/10.5281/zenodo.19410504)

### License
AGPL-3.0 with Academic Additional Permission. Free for research (citation required).

**Contact**: mark@dillerop.com
                """)

    # Launch
    print(f"🧠 EgoRA v{__version__} — Launching Gradio demo...")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


def cmd_version(args):
    """Print EgoRA version."""
    from egora._version import __version__
    print(f"egora {__version__}")


def main():
    parser = argparse.ArgumentParser(
        prog="egora",
        description="EgoRA — Entropy-Governed Orthogonality Regularization for Adaptation",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Fine-tune a model with EgoRA")
    p_train.add_argument("model", help="HF Hub model name or local path")
    p_train.add_argument("dataset", help="HF Hub dataset name or local path")
    p_train.add_argument("-o", "--output-dir", default="./egora-output", help="Output directory (default: ./egora-output)")
    p_train.add_argument("-r", "--rank", type=int, default=16, help="LoRA rank (default: 16)")
    p_train.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    p_train.add_argument("--adapter-type", default="egora", choices=["egora", "dora", "rslora"], help="Adapter type (default: egora)")
    p_train.add_argument("--no-egora", action="store_true", help="Disable EgoRA penalty (plain LoRA)")
    p_train.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs (default: 3)")
    p_train.add_argument("--batch-size", type=int, default=4, help="Per-device batch size (default: 4)")
    p_train.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    p_train.add_argument("--max-length", type=int, default=512, help="Max token sequence length (default: 512)")
    p_train.add_argument("--max-samples", type=int, default=None, help="Max training samples (default: all)")
    p_train.add_argument("--text-column", default="text", help="Dataset text column name (default: text, auto-detected)")
    p_train.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    p_train.add_argument("--shadow-refresh", type=int, default=100, help="Shadow refresh interval in steps (default: 100)")
    p_train.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    p_train.add_argument("--logging-steps", type=int, default=10, help="Log every N steps (default: 10)")
    p_train.add_argument("--fp32", action="store_true", help="Force fp32 (default: bf16 on CUDA)")
    p_train.add_argument("--save", action="store_true", help="Save adapter checkpoints")
    p_train.add_argument("--merge", action="store_true", help="Merge adapters and save full model")
    p_train.add_argument("--skip-diagnose", action="store_true", help="Skip post-training diagnostics")
    # Entropy governor parameters
    p_train.add_argument("--egora-alpha", type=float, default=1.359, help="Entropy governor alpha coefficient (default: e/2 ≈ 1.359)")
    p_train.add_argument("--egora-lam-floor", type=float, default=0.6931, help="Minimum lambda floor (default: ln(2) ≈ 0.6931)")
    # Adaptive alpha parameters
    p_train.add_argument("--adaptive-alpha", action="store_true", help="Enable adaptive alpha (adjusts penalty based on generalization gap)")
    p_train.add_argument("--alpha-min", type=float, default=0.05, help="Minimum adaptive alpha (default: 0.05)")
    p_train.add_argument("--alpha-max", type=float, default=3.0, help="Maximum adaptive alpha (default: 3.0)")
    p_train.add_argument("--alpha-gamma", type=float, default=0.15, help="Adaptive alpha learning rate (default: 0.15)")
    p_train.add_argument("--alpha-cooldown", action="store_true", help="Enable alpha cooldown when gap stabilizes")
    p_train.set_defaults(func=cmd_train)

    # ── diagnose ─────────────────────────────────────────────────────────────
    p_diag = subparsers.add_parser("diagnose", help="Compute rotation geometry between base and tuned model")
    p_diag.add_argument("base_model", help="Path or HF Hub name for the base model")
    p_diag.add_argument("tuned_model", help="Path or HF Hub name for the tuned model")
    p_diag.add_argument("-o", "--output", default=None, help="Output JSON path (default: stdout only)")
    p_diag.add_argument("-t", "--target-modules", default=None, help="Comma-separated projection names (default: q_proj,k_proj,v_proj,o_proj)")
    p_diag.add_argument("--damage-threshold", type=float, default=5.0, help="Rotation threshold for damage classification (degrees, default: 5.0)")
    p_diag.add_argument("--diversity", action="store_true", help="Also compute interhead diversity matrix")
    p_diag.add_argument("--plot", action="store_true", help="Generate rotation report plot (requires matplotlib)")
    p_diag.set_defaults(func=cmd_diagnose)

    # ── info ─────────────────────────────────────────────────────────────────
    p_info = subparsers.add_parser("info", help="Show model architecture info for EgoRA")
    p_info.add_argument("model", help="Path or HF Hub name")
    p_info.add_argument("-t", "--target-modules", default=None, help="Comma-separated projection names")
    p_info.set_defaults(func=cmd_info)

    # ── demo ─────────────────────────────────────────────────────────────────
    p_demo = subparsers.add_parser("demo", help="Launch Gradio web demo for diagnostics")
    p_demo.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    p_demo.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    p_demo.add_argument("--share", action="store_true", help="Create public Gradio share link")
    p_demo.set_defaults(func=cmd_demo)

    # ── version ──────────────────────────────────────────────────────────────
    p_ver = subparsers.add_parser("version", help="Print EgoRA version")
    p_ver.set_defaults(func=cmd_version)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
