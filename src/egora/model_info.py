"""
EgoRA Model Info
================
Architecture facts relevant to EgoRA: attention head geometry, the
critical rotation angle θ_crit = arcsin(1/√d_head), and exact LoRA
parameter counts (GQA-aware).
"""

import math
from typing import Any, Dict, List, Optional

import torch.nn as nn

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def theta_crit_deg(head_dim: int) -> float:
    """Critical rotation angle θ_crit = arcsin(1/√d_head), in degrees."""
    if head_dim is None or head_dim < 1:
        raise ValueError(f"head_dim must be a positive integer, got {head_dim!r}")
    return math.degrees(math.asin(1.0 / math.sqrt(head_dim)))


def _text_config(config):
    """Return the text sub-config for multimodal configs, else the config itself."""
    if config is None:
        return None
    if hasattr(config, "get_text_config"):
        try:
            return config.get_text_config()
        except Exception:
            pass
    return getattr(config, "text_config", None) or config


def get_attention_dims(model_or_config) -> Dict[str, Optional[int]]:
    """Read attention dimensions from a model or HF config.

    Returns:
        Dict with ``head_dim``, ``num_attention_heads``, ``num_key_value_heads``,
        ``hidden_size`` and ``num_layers`` (``None`` when unknown).
    """
    config = getattr(model_or_config, "config", model_or_config)
    config = _text_config(config)

    def _get(*names):
        for n in names:
            v = getattr(config, n, None)
            if isinstance(v, int):
                return v
        return None

    hidden = _get("hidden_size", "n_embd", "d_model")
    n_heads = _get("num_attention_heads", "n_head", "num_heads")
    n_kv = _get("num_key_value_heads", "num_kv_heads", "n_kv_heads") or n_heads
    head_dim = _get("head_dim")
    if head_dim is None and hidden and n_heads:
        head_dim = hidden // n_heads
    n_layers = _get("num_hidden_layers", "n_layer", "num_layers")

    return {
        "head_dim": head_dim,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "hidden_size": hidden,
        "num_layers": n_layers,
    }


def _target_linears(model: nn.Module, target_modules: List[str]):
    for name, module in model.named_modules():
        attr = name.rsplit(".", 1)[-1]
        if attr in target_modules and isinstance(module, nn.Linear):
            yield name, attr, module


def summarize_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    ranks=(8, 16, 32, 64),
) -> Dict[str, Any]:
    """Summarize a model's architecture for EgoRA.

    Works on models instantiated on the ``meta`` device (no weights needed).

    LoRA estimates use each target layer's real shape: ``r * (in + out)``.
    """
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    total_params = sum(p.numel() for p in model.parameters())
    target_params = 0
    target_count = 0
    lora_per_rank_unit = 0
    layers = set()
    shapes: Dict[str, tuple] = {}

    for name, attr, module in _target_linears(model, target_modules):
        target_count += 1
        target_params += module.weight.numel()
        lora_per_rank_unit += module.in_features + module.out_features
        shapes.setdefault(attr, (module.out_features, module.in_features))
        for part in name.split("."):
            if part.isdigit():
                layers.add(int(part))
                break

    dims = get_attention_dims(model)
    head_dim = dims["head_dim"]

    lora_estimates = {
        int(r): {
            "params": int(r * lora_per_rank_unit),
            "percent": (r * lora_per_rank_unit / total_params * 100) if total_params else 0.0,
        }
        for r in ranks
    }

    return {
        "total_params": int(total_params),
        "target_modules": list(target_modules),
        "target_count": target_count,
        "target_params": int(target_params),
        "num_layers": len(layers) if layers else dims["num_layers"],
        "head_dim": head_dim,
        "num_attention_heads": dims["num_attention_heads"],
        "num_key_value_heads": dims["num_key_value_heads"],
        "hidden_size": dims["hidden_size"],
        "theta_crit_deg": theta_crit_deg(head_dim) if head_dim else None,
        "projection_shapes": {k: list(v) for k, v in shapes.items()},
        "lora_estimates": lora_estimates,
    }


def load_model_skeleton(model_name: str, trust_remote_code: bool = False):
    """Instantiate a HF causal LM on the ``meta`` device (config only, no weights)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    return model


def format_model_info_markdown(name: str, info: Dict[str, Any]) -> str:
    """Render :func:`summarize_model` output as a Markdown report."""
    lines = [
        f"## Model Info: {name}",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Total params | {info['total_params']:,} |",
        f"| Layers | {info['num_layers']} |",
        f"| Target projections | {info['target_count']} |",
    ]
    if info["total_params"]:
        pct = info["target_params"] / info["total_params"] * 100
        lines.append(f"| Target params | {info['target_params']:,} ({pct:.1f}%) |")
    if info.get("num_attention_heads"):
        lines.append(
            f"| Attention heads (Q / KV) | {info['num_attention_heads']} / "
            f"{info['num_key_value_heads']} |"
        )
    if info.get("head_dim"):
        lines.append(f"| d_head | {info['head_dim']} |")
        lines.append(f"| θ_crit = arcsin(1/√d_head) | {info['theta_crit_deg']:.2f}° |")
    lines += [
        "",
        "### LoRA Parameter Estimates",
        "",
        "| Rank | Trainable Params | % of Total |",
        "|------|-----------------|------------|",
    ]
    for r, est in info["lora_estimates"].items():
        lines.append(f"| {r} | {est['params']:,} | {est['percent']:.2f}% |")
    return "\n".join(lines) + "\n"
