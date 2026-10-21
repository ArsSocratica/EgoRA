"""
EgoRA Command-Line Interface
=============================
Provides ``egora`` CLI commands for diagnostics and quick experimentation.

Commands:
  egora train MODEL DATASET                — Fine-tune a model with EgoRA
  egora diagnose BASE_MODEL TUNED_MODEL    — Compute rotation geometry (TUNED may be an adapter)
  egora info MODEL                         — Show model architecture summary
  egora demo                               — Launch Gradio web demo
  egora version                            — Print version
"""

import argparse
import os
import sys


def _require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("Error: `transformers` is required for model loading.")
        print("Install with: pip install transformers")
        sys.exit(1)


def _load_model(path_or_name: str, trust_remote_code: bool = True):
    """Load a HuggingFace model from path or Hub name (stored dtype, CPU)."""
    _require_transformers()
    from egora.loading import load_causal_lm

    print(f"Loading model: {path_or_name} ...")
    return load_causal_lm(path_or_name, trust_remote_code=trust_remote_code)


def cmd_diagnose(args):
    """Run rotation geometry diagnostics between base and tuned model."""
    from egora.geometry import compute_head_geometry, compute_interhead_diversity, save_geometry
    from egora.loading import load_tuned_model
    from egora.model_info import get_attention_dims, theta_crit_deg

    base_model = _load_model(args.base_model)
    print(f"Loading fine-tuned model: {args.tuned_model} ...")
    tuned_model, kind = load_tuned_model(
        base_model, args.tuned_model, trust_remote_code=True
    )
    if kind != "model":
        print(f"  Detected {kind.replace('_', ' ')}: merged into a copy of the base model")

    target_modules = args.target_modules.split(",") if args.target_modules else None

    head_dim = args.head_dim or get_attention_dims(base_model)["head_dim"]
    model_crit = theta_crit_deg(head_dim) if head_dim else None
    verdict_crit = 5.0 if args.fixed_threshold or model_crit is None else model_crit
    damage_threshold = args.damage_threshold if args.damage_threshold is not None else verdict_crit

    print(f"Computing {args.granularity}-level geometry ...")
    results = compute_head_geometry(
        base_model, tuned_model,
        target_modules=target_modules,
        damage_threshold_deg=damage_threshold,
        granularity=args.granularity,
        head_dim=head_dim,
        verdict_theta_crit=verdict_crit,
    )

    if "error" in results:
        print(f"Error: {results['error']}")
        sys.exit(1)

    # Print summary
    print()
    print("=" * 60)
    print("  EgoRA Rotation Geometry Report")
    print("=" * 60)
    unit = "Heads" if results["granularity"] == "head" else "Projections"
    print(f"  {unit + ' analyzed:':22s}{results['n_heads']}")
    if results.get("theta_crit_deg"):
        print(f"  d_head / θ_crit:      {results['head_dim']} / {results['theta_crit_deg']:.2f}°")
    print(f"  Damage threshold:     {results['damage_threshold_deg']:.2f}°")
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
    verdict = results["verdict"]["message"]
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
    _require_transformers()
    from egora.model_info import load_model_skeleton, summarize_model

    target_modules = args.target_modules.split(",") if args.target_modules else None
    try:
        model = load_model_skeleton(args.model, trust_remote_code=True)
    except Exception:
        model = _load_model(args.model)
    info = summarize_model(model, target_modules=target_modules)

    print()
    print("=" * 60)
    print("  EgoRA Model Info")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Total params:   {info['total_params']:,}")
    print()
    print(f"  Target modules: {', '.join(info['target_modules'])}")
    print(f"  Target layers:  {info['num_layers']}")
    pct = info["target_params"] / info["total_params"] * 100 if info["total_params"] else 0
    print(f"  Target params:  {info['target_params']:,} ({pct:.1f}%)")
    print(f"  Projections:    {info['target_count']}")
    if info["num_attention_heads"]:
        print(f"  Heads (Q / KV): {info['num_attention_heads']} / {info['num_key_value_heads']}")
    if info["head_dim"]:
        print(f"  d_head:         {info['head_dim']}")
        print(f"  θ_crit:         {info['theta_crit_deg']:.2f}° (arcsin(1/√d_head))")
    print()

    for rank, est in info["lora_estimates"].items():
        print(f"  LoRA rank={rank:2d}: {est['params']:>12,} trainable params "
              f"({est['percent']:.2f}%)")

    print()
    print("=" * 60)


def cmd_train(args):
    """Fine-tune a model with EgoRA in one command."""
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: `transformers` and `datasets` are required for training.")
        print("Install with: pip install transformers datasets accelerate")
        sys.exit(1)

    import torch

    from egora.peft_compat import EgoRALoraConfig
    from egora.trainer import EgoRATrainer, EgoRATrainingArguments

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

    # Detect best available device
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    use_fp32 = args.fp32

    if has_cuda and not use_fp32:
        mixed_precision = {"bf16": True}
        model_dtype = torch.bfloat16
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif has_mps:
        # MPS accelerates fp32 natively; fp16 causes NaN in softmax/log
        mixed_precision = {}
        model_dtype = torch.float32
        device_info = "Apple MPS (Metal)"
    else:
        mixed_precision = {}
        model_dtype = torch.float32
        device_info = "CPU"

    print(f"Device: {device_info}")

    train_args = EgoRATrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_strategy="epoch" if args.save else "no",
        gradient_accumulation_steps=args.gradient_accumulation,
        extra_args=mixed_precision,
    )

    model_kwargs = {}
    if model_dtype != torch.float32:
        model_kwargs["torch_dtype"] = model_dtype

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
            print(f"  Damaged projections: {report['damaged_fraction']*100:.1f}%")
            print(f"  Verdict: {report['verdict']['message']}")
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
        import gradio  # noqa: F401
    except ImportError:
        print("Error: gradio is required for the demo.")
        print("Install with: pip install gradio")
        sys.exit(1)

    from egora._version import __version__
    from egora.demo import build_demo, launch_demo

    print(f"EgoRA v{__version__} — Launching Gradio demo...")
    demo = build_demo(trust_remote_code=args.trust_remote_code)
    launch_demo(demo, server_name=args.host, server_port=args.port, share=args.share)


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
    p_diag.add_argument("--damage-threshold", type=float, default=None, help="Rotation threshold for damage classification (degrees, default: the verdict θ_crit)")
    p_diag.add_argument("--granularity", choices=["projection", "head"], default="projection", help="Measure whole projections (default) or individual attention heads")
    p_diag.add_argument("--head-dim", type=int, default=None, help="Attention head dimension (default: read from model config)")
    p_diag.add_argument("--fixed-threshold", action="store_true", help="Use the fixed 5°/3° verdict thresholds instead of the model-specific θ_crit")
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
    p_demo.add_argument("--trust-remote-code", action="store_true", help="Allow models with custom code (only for models you trust)")
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
