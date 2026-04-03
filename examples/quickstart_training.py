"""
EgoRA Quick Start — Fine-tuning with Entropy-Governed Regularization
=====================================================================

This example shows the minimal training loop with EgoRA.
Requires: pip install egora transformers datasets torch

Usage:
    python quickstart_training.py --model meta-llama/Llama-3.2-1B \
        --dataset wikitext --rank 16 --epochs 1
"""

import argparse
import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="EgoRA fine-tuning example")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B", help="HF model name")
    parser.add_argument("--dataset", default="wikitext", help="HF dataset name")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--shadow-refresh", type=int, default=100, help="Shadow refresh interval")
    parser.add_argument("--output-dir", default="./egora-output", help="Output directory")
    parser.add_argument("--use-egora", action="store_true", default=True, help="Enable EgoRA penalty")
    parser.add_argument("--no-egora", action="store_true", help="Disable EgoRA (baseline LoRA)")
    args = parser.parse_args()

    use_egora = args.use_egora and not args.no_egora

    # ── Load model ───────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # ── Apply EgoRA LoRA adapters ────────────────────────────────────────────
    from egora import apply_lora, get_total_egora_penalty, refresh_all_shadows
    from egora import EntropyGovernor

    print(f"Applying LoRA (rank={args.rank}, egora={use_egora})")
    lora_modules = apply_lora(
        model, rank=args.rank, lora_alpha=int(args.alpha),
        use_egora=use_egora,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    governor = EntropyGovernor(alpha=1.359, lam_floor=0.6931)

    # ── Load dataset ─────────────────────────────────────────────────────────
    from datasets import load_dataset

    print(f"Loading dataset: {args.dataset}/{args.dataset_config}")
    dataset = load_dataset(args.dataset, args.dataset_config, split="train")

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=args.max_length,
            padding="max_length", return_tensors="pt",
        )

    dataset = dataset.filter(lambda x: len(x["text"]) > 50)
    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format("torch")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ── Training loop ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    device = next(model.parameters()).device
    model.train()

    print(f"\nTraining for {args.epochs} epoch(s)...")
    for epoch in range(args.epochs):
        total_loss_sum = 0
        penalty_sum = 0
        n_steps = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            task_loss = outputs.loss

            # EgoRA penalty
            if use_egora:
                lam = governor.compute_lambda(outputs.logits)
                penalty = get_total_egora_penalty(lora_modules)
                total_loss = task_loss + lam * penalty
                penalty_sum += penalty.item()
            else:
                total_loss = task_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()

            total_loss_sum += total_loss.item()
            n_steps += 1

            # Shadow refresh
            if use_egora and step > 0 and step % args.shadow_refresh == 0:
                refresh_all_shadows(lora_modules, momentum=0.9)
                print(f"  [step {step}] Shadow refreshed")

            if step % 50 == 0:
                avg_loss = total_loss_sum / n_steps
                avg_pen = penalty_sum / n_steps if use_egora else 0
                print(f"  Epoch {epoch+1} Step {step}: loss={avg_loss:.4f} penalty={avg_pen:.4f}")

            if step >= 500:  # Cap for demo
                break

        print(f"Epoch {epoch+1} done. Avg loss: {total_loss_sum/n_steps:.4f}")

    # ── Post-training diagnostics ────────────────────────────────────────────
    print("\nRunning post-training diagnostics...")
    from egora import compute_head_geometry

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map="cpu",
    )

    model_cpu = model.cpu().float()
    results = compute_head_geometry(base_model, model_cpu)

    print(f"  Mean rotation: {results['theta_bar_deg']:.2f}°")
    print(f"  Damaged heads: {results['damaged_fraction']*100:.1f}%")
    print(f"  Mode distribution: {results['mode_counts']}")

    if results['theta_bar_deg'] < 3.0:
        print("  ✅ SAFE — Knowledge well preserved")
    elif results['theta_bar_deg'] < 5.0:
        print("  ⚠️  WARNING — Check benchmarks")
    else:
        print("  ❌ DANGER — Significant rotation detected")

    # Save
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    from egora.geometry import save_geometry
    save_geometry(results, os.path.join(args.output_dir, "geometry.json"))
    print(f"\nResults saved to {args.output_dir}/")

    # Optional plot
    try:
        from egora.plotting import plot_rotation_report
        fig = plot_rotation_report(results)
        fig.savefig(os.path.join(args.output_dir, "report.png"), dpi=150, bbox_inches="tight")
        print(f"Report plot saved to {args.output_dir}/report.png")
    except ImportError:
        print("(matplotlib not installed, skipping plot)")


if __name__ == "__main__":
    main()
