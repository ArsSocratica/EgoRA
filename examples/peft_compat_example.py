"""
EgoRA + PEFT Compatibility Example
====================================

Shows how to use the EgoRAPeftModel wrapper for a PEFT-like API.

Requires: pip install egora transformers

Usage:
    python peft_compat_example.py --model meta-llama/Llama-3.2-1B
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="EgoRA PEFT compatibility example")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B", help="HF model name")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--output-dir", default="./egora-peft-output", help="Output directory")
    args = parser.parse_args()

    from egora.peft_compat import EgoRAPeftModel, EgoRALoraConfig

    # ── Configure ────────────────────────────────────────────────────────────
    config = EgoRALoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        use_egora=True,
        egora_alpha=1.359,
        egora_lam_floor=0.6931,
        shadow_refresh_interval=100,
    )

    # ── Load model with EgoRA ────────────────────────────────────────────────
    print(f"Loading {args.model} with EgoRA (rank={args.rank})...")
    model = EgoRAPeftModel.from_pretrained(
        args.model,
        config,
        model_kwargs={
            "torch_dtype": torch.float32,
            "device_map": "cpu",
        },
    )

    model.print_trainable_parameters()

    # ── Simulate training ────────────────────────────────────────────────────
    print("\nSimulating training step...")
    dummy_input = torch.randint(0, 1000, (1, 64))
    outputs = model(input_ids=dummy_input, labels=dummy_input)

    # One-liner loss computation with EgoRA
    total_loss = model.compute_total_loss(outputs.loss, outputs.logits)
    print(f"  Task loss:  {outputs.loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Lambda:     {model.compute_lambda(outputs.logits):.4f}")

    # ── Save adapter ─────────────────────────────────────────────────────────
    print(f"\nSaving to {args.output_dir}...")
    model.save_pretrained(args.output_dir)

    print(f"Saved files:")
    import os
    for f in os.listdir(args.output_dir):
        size = os.path.getsize(os.path.join(args.output_dir, f))
        print(f"  {f} ({size:,} bytes)")

    # ── Reload ───────────────────────────────────────────────────────────────
    print("\nReloading adapter...")
    model2 = EgoRAPeftModel.from_pretrained(
        args.model, config,
        model_kwargs={"torch_dtype": torch.float32, "device_map": "cpu"},
    )
    model2.load_adapter(args.output_dir)
    print("Adapter reloaded successfully.")

    # ── Merge and unload ─────────────────────────────────────────────────────
    print("\nMerging LoRA into base weights...")
    base = model2.merge_and_unload()
    print(f"Merged model type: {type(base).__name__}")
    print("Done!")


if __name__ == "__main__":
    main()
