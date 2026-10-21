"""
EgoRA Model Loading
===================
Load base models and fine-tuned models for diagnostics. The fine-tuned
reference can be a full model, a PEFT LoRA adapter, or an EgoRA adapter
directory (``egora_config.json`` + ``adapter_model.bin``).
"""

import copy
import os
from typing import Any, Optional

import torch


def _resolve_dtype(dtype: Any):
    if dtype is None or dtype == "auto":
        return "auto"
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


def load_causal_lm(
    name_or_path: str,
    dtype: Any = "auto",
    device: str = "cpu",
    trust_remote_code: bool = False,
):
    """Load a HuggingFace causal LM for analysis (eval mode)."""
    from transformers import AutoModelForCausalLM

    from egora._compat import dtype_kwargs

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        device_map=device,
        trust_remote_code=trust_remote_code,
        **dtype_kwargs(_resolve_dtype(dtype)),
    )
    model.eval()
    return model


def _has_file(ref: str, filename: str) -> bool:
    if os.path.isdir(ref):
        return os.path.exists(os.path.join(ref, filename))
    if os.path.exists(ref):
        return False
    try:
        from huggingface_hub import file_exists

        return file_exists(ref, filename)
    except Exception:
        return False


def detect_reference_kind(ref: str) -> str:
    """Return ``"egora_adapter"``, ``"peft_adapter"`` or ``"model"``."""
    if _has_file(ref, "egora_config.json"):
        return "egora_adapter"
    if _has_file(ref, "adapter_config.json"):
        return "peft_adapter"
    return "model"


def _download_dir(ref: str) -> str:
    if os.path.isdir(ref):
        return ref
    from huggingface_hub import snapshot_download

    return snapshot_download(ref)


def merge_egora_adapter(base_model: torch.nn.Module, adapter_dir: str) -> torch.nn.Module:
    """Return a copy of ``base_model`` with an EgoRA adapter merged in."""
    from egora.adapters import apply_lora, merge_lora
    from egora.peft_compat import EgoRALoraConfig

    config = EgoRALoraConfig.load(os.path.join(adapter_dir, "egora_config.json"))
    model = copy.deepcopy(base_model)
    modules = apply_lora(
        model,
        rank=config.r,
        lora_alpha=config.lora_alpha,
        use_egora=False,
        target_modules=config.target_modules,
        adapter_type=config.adapter_type,
    )
    state = torch.load(
        os.path.join(adapter_dir, "adapter_model.bin"), map_location="cpu", weights_only=True
    )
    with torch.no_grad():
        for i, m in enumerate(modules):
            m.lora_A.copy_(state[f"adapter.{i}.lora_A"])
            m.lora_B.copy_(state[f"adapter.{i}.lora_B"])
    merge_lora(model, modules)
    return model


def merge_peft_adapter(base_model: torch.nn.Module, adapter_ref: str) -> torch.nn.Module:
    """Return a copy of ``base_model`` with a PEFT adapter merged in."""
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("Loading PEFT adapters requires `pip install peft`.")
    model = PeftModel.from_pretrained(copy.deepcopy(base_model), adapter_ref)
    return model.merge_and_unload()


def load_tuned_model(
    base_model: torch.nn.Module,
    tuned_ref: str,
    dtype: Any = "auto",
    device: str = "cpu",
    trust_remote_code: bool = False,
    kind: Optional[str] = None,
):
    """Load the fine-tuned counterpart of ``base_model``.

    Adapters are merged into a copy of ``base_model``, so the base is only
    loaded once.

    Returns:
        ``(tuned_model, kind)``.
    """
    kind = kind or detect_reference_kind(tuned_ref)
    if kind == "egora_adapter":
        model = merge_egora_adapter(base_model, _download_dir(tuned_ref))
    elif kind == "peft_adapter":
        model = merge_peft_adapter(base_model, tuned_ref)
    else:
        model = load_causal_lm(tuned_ref, dtype, device, trust_remote_code)
    model.eval()
    return model, kind
