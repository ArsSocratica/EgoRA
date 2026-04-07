"""
EgoRA — Entropy-Governed Orthogonality Regularization for Adaptation
=====================================================================

EgoRA is a dynamic, information-theoretic regularization method for
fine-tuning neural networks. It uses the model's own output entropy to
modulate an orthogonality penalty on LoRA adapter weights, preventing
knowledge destruction and rank collapse during adaptation.

Quick start::

    import torch
    from transformers import AutoModelForCausalLM
    from egora import apply_lora, get_total_egora_penalty, refresh_all_shadows
    from egora import EntropyGovernor

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    lora_modules = apply_lora(model, rank=16, use_egora=True)
    gov = EntropyGovernor(alpha=1.359, lam_floor=0.6931)

    for batch in dataloader:
        outputs = model(**batch)
        task_loss = outputs.loss

        lam = gov.compute_lambda(outputs.logits)
        penalty = get_total_egora_penalty(lora_modules)
        total_loss = task_loss + lam * penalty

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Periodically refresh shadows:
    refresh_all_shadows(lora_modules, momentum=0.9)

Post-training diagnostics::

    from egora import compute_head_geometry

    results = compute_head_geometry(base_model, tuned_model)
    print(f"Mean rotation: {results['theta_bar_deg']:.2f} deg")
    print(f"Damaged heads: {results['damaged_fraction']*100:.1f}%")
"""

from egora._version import __version__

# ── Adapters ──────────────────────────────────────────────────────────────────
from egora.adapters import (
    EgoRALoRALinear,
    DoRALinear,
    rsLoRALinear,
    ADAPTER_CLASSES,
    apply_lora,
    get_total_egora_penalty,
    refresh_all_shadows,
    merge_lora,
    unmerge_lora,
)

# ── Governor ──────────────────────────────────────────────────────────────────
from egora.governor import EntropyGovernor, GovernorConfig

# ── Geometry / Diagnostics ────────────────────────────────────────────────────
from egora.geometry import (
    compute_head_geometry,
    compute_interhead_diversity,
    save_geometry,
)

# ── PEFT Compatibility ───────────────────────────────────────────────────────
from egora.peft_compat import EgoRAPeftModel, EgoRALoraConfig

# ── Trainer Integration (lazy — requires transformers) ──────────────────────
def __getattr__(name):
    if name in ("EgoRATrainer", "EgoRATrainingArguments"):
        from egora.trainer import EgoRATrainer, EgoRATrainingArguments
        globals()["EgoRATrainer"] = EgoRATrainer
        globals()["EgoRATrainingArguments"] = EgoRATrainingArguments
        return globals()[name]
    raise AttributeError(f"module 'egora' has no attribute {name!r}")

__all__ = [
    # Version
    "__version__",
    # Adapters
    "EgoRALoRALinear",
    "DoRALinear",
    "rsLoRALinear",
    "ADAPTER_CLASSES",
    "apply_lora",
    "get_total_egora_penalty",
    "refresh_all_shadows",
    "merge_lora",
    "unmerge_lora",
    # Governor
    "EntropyGovernor",
    "GovernorConfig",
    # Geometry
    "compute_head_geometry",
    "compute_interhead_diversity",
    "save_geometry",
    # PEFT Compatibility
    "EgoRAPeftModel",
    "EgoRALoraConfig",
    # Trainer
    "EgoRATrainer",
    "EgoRATrainingArguments",
]
