"""
EgoRA Command-Line Interface
=============================
Provides ``egora`` CLI commands for diagnostics and quick experimentation.

Commands:
  egora diagnose BASE_MODEL TUNED_MODEL   — Compute rotation geometry
  egora info MODEL                         — Show model architecture summary
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
