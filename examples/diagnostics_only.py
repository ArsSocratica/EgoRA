"""
EgoRA Diagnostics — Compare Base vs Fine-Tuned Model
=====================================================

Standalone script to analyze rotation geometry between any two model
checkpoints. No training needed — just point at two model directories.

Requires: pip install egora[diagnostics] transformers

Usage:
    python diagnostics_only.py \
        --base meta-llama/Llama-3.2-1B \
        --tuned ./my-finetuned-model \
        --output ./analysis
"""

import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser(description="EgoRA rotation diagnostics")
    parser.add_argument("--base", required=True, help="Base model path or HF name")
    parser.add_argument("--tuned", required=True, help="Tuned model path or HF name")
    parser.add_argument("--output", default="./egora-analysis", help="Output directory")
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated projection names")
    parser.add_argument("--damage-threshold", type=float, default=5.0,
                        help="Damage threshold in degrees")
    parser.add_argument("--diversity", action="store_true",
                        help="Compute interhead diversity matrix")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM
    from egora.geometry import compute_head_geometry, compute_interhead_diversity, save_geometry

    target_modules = args.target_modules.split(",")

    # Load models
    print(f"Loading base model: {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True,
    )

    print(f"Loading tuned model: {args.tuned}")
    tuned_model = AutoModelForCausalLM.from_pretrained(
        args.tuned, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True,
    )

    # Compute geometry
    print("Computing rotation geometry ...")
    results = compute_head_geometry(
        base_model, tuned_model,
        target_modules=target_modules,
        damage_threshold_deg=args.damage_threshold,
    )

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    # Print summary
    print()
    print(f"  Heads:          {results['n_heads']}")
    print(f"  Mean rotation:  {results['theta_bar_deg']:.2f}° ± {results['theta_std_deg']:.2f}°")
    print(f"  Range:          {results['theta_min_deg']:.2f}° — {results['theta_max_deg']:.2f}°")
    print(f"  Damaged:        {results['damaged_fraction']*100:.1f}%")
    print()
    print("  Modes:", {m: f"{f*100:.1f}%" for m, f in results['mode_fractions'].items()})
    print()

    # Diversity
    if args.diversity:
        print("Computing interhead diversity ...")
        import numpy as np
        sim_matrix, names = compute_interhead_diversity(tuned_model, target_modules=target_modules)
        if sim_matrix is not None:
            off_diag = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
            print(f"  Mean similarity: {off_diag.mean():.4f}")
            print(f"  Diversity score: {1 - off_diag.mean():.4f}")
            results['interhead_diversity'] = sim_matrix.tolist()
        print()

    # Save
    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "geometry.json")
    save_geometry(results, json_path)
    print(f"Results saved to {json_path}")

    # Plot
    try:
        from egora.plotting import plot_rotation_report
        fig = plot_rotation_report(results)
        plot_path = os.path.join(args.output, "rotation_report.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("(Install egora[diagnostics] for plots)")

    # Verdict
    theta = results['theta_bar_deg']
    if theta < 3.0:
        print("\n✅ SAFE — Knowledge well preserved")
    elif theta < 5.0:
        print("\n⚠️  WARNING — Moderate rotation, check benchmarks")
    else:
        print("\n❌ DANGER — Significant knowledge loss expected")


if __name__ == "__main__":
    main()
