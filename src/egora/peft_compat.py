"""
HuggingFace PEFT Compatibility Layer
======================================
Provides helpers to convert between EgoRA adapters and PEFT-compatible
formats, enabling interoperability with the HuggingFace ecosystem.

Features:
  - Convert EgoRA LoRA config to PEFT LoraConfig
  - Wrap a PEFT model with EgoRA entropy governor
  - Save/load EgoRA adapters in PEFT-compatible format
  - Inject EgoRA penalty into PEFT training loops

Usage::

    from egora.peft_compat import EgoRAPeftModel, EgoRALoraConfig

    config = EgoRALoraConfig(r=16, lora_alpha=32, use_egora=True)
    model = EgoRAPeftModel.from_pretrained("meta-llama/Llama-3.1-8B", config)

    # Training loop with PEFT + EgoRA
    model.train()
    loss = model.compute_total_loss(task_loss, logits)
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn


@dataclass
class EgoRALoraConfig:
    """Configuration for EgoRA-enhanced LoRA adapters.

    Compatible with key PEFT LoraConfig parameters, with additional
    EgoRA-specific fields.

    Attributes:
        r: LoRA rank.
        lora_alpha: LoRA scaling numerator.
        lora_dropout: Dropout on LoRA path.
        target_modules: Module names to replace with LoRA.
        use_egora: Enable EgoRA entropy-governed penalty.
        egora_alpha: Entropy governor alpha (default: e/2).
        egora_lam_floor: Minimum lambda (default: ln(2)).
        egora_adaptive_alpha: Enable adaptive alpha.
        adapter_type: One of 'egora', 'dora', 'rslora'.
        shadow_momentum: EMA momentum for shadow refresh.
        shadow_refresh_interval: Steps between shadow refreshes.
    """
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    use_egora: bool = True
    egora_alpha: float = 1.359
    egora_lam_floor: float = 0.6931
    egora_adaptive_alpha: bool = False
    adapter_type: str = "egora"
    shadow_momentum: float = 0.9
    shadow_refresh_interval: int = 100

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        """Save config to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EgoRALoraConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_peft_config(self):
        """Convert to a PEFT LoraConfig (requires peft installed).

        Returns:
            A ``peft.LoraConfig`` instance with matching LoRA parameters.
        """
        try:
            from peft import LoraConfig
        except ImportError:
            raise ImportError(
                "peft is required for to_peft_config(). "
                "Install with: pip install peft"
            )

        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
        )


class EgoRAPeftModel(nn.Module):
    """Wrapper that adds EgoRA entropy-governed penalty to any model.

    Can wrap either:
      - A plain HuggingFace model (applies EgoRA adapters internally)
      - A PEFT-wrapped model (adds entropy governance on top)

    Args:
        model: A PyTorch model (HuggingFace or PEFT-wrapped).
        config: An :class:`EgoRALoraConfig` instance.
    """

    def __init__(self, model: nn.Module, config: EgoRALoraConfig):
        super().__init__()
        self.model = model
        self.config = config
        self._step_count = 0

        from egora.governor import EntropyGovernor, GovernorConfig
        self.governor = EntropyGovernor(GovernorConfig(
            alpha=config.egora_alpha,
            lam_floor=config.egora_lam_floor,
            adaptive_alpha=config.egora_adaptive_alpha,
        ))

        # Check if model already has PEFT adapters
        self._is_peft = hasattr(model, 'peft_config') or hasattr(model, 'get_base_model')
        self._lora_modules = []

        if not self._is_peft:
            from egora.adapters import apply_lora
            self._lora_modules = apply_lora(
                model,
                rank=config.r,
                lora_alpha=config.lora_alpha,
                use_egora=config.use_egora,
                target_modules=config.target_modules,
                lora_dropout=config.lora_dropout,
                adapter_type=config.adapter_type,
            )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: EgoRALoraConfig,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "EgoRAPeftModel":
        """Load a pretrained model and apply EgoRA adapters.

        Args:
            model_name_or_path: HuggingFace model name or local path.
            config: EgoRA configuration.
            model_kwargs: Additional kwargs for model loading.

        Returns:
            An :class:`EgoRAPeftModel` instance.
        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError("transformers is required. Install with: pip install transformers")

        kwargs = model_kwargs or {}
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        return cls(model, config)

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)

    def compute_egora_penalty(self) -> torch.Tensor:
        """Compute the EgoRA orthogonality penalty.

        Returns:
            Scalar penalty tensor (differentiable).
        """
        if self._lora_modules:
            from egora.adapters import get_total_egora_penalty
            return get_total_egora_penalty(self._lora_modules)
        return torch.tensor(0.0)

    def compute_lambda(self, logits: torch.Tensor) -> float:
        """Compute entropy-governed scaling coefficient.

        Args:
            logits: Model output logits.

        Returns:
            Scalar lambda value.
        """
        return self.governor.compute_lambda(logits)

    def compute_total_loss(
        self, task_loss: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute total loss = task_loss + lambda * EgoRA_penalty.

        This is the main method for training loops.

        Args:
            task_loss: The task-specific loss (e.g., cross-entropy).
            logits: Model output logits for entropy computation.

        Returns:
            Total loss tensor (differentiable).
        """
        lam = self.compute_lambda(logits)
        penalty = self.compute_egora_penalty()
        total = task_loss + lam * penalty

        self._step_count += 1
        if (self.config.shadow_refresh_interval > 0
                and self._step_count % self.config.shadow_refresh_interval == 0):
            self.refresh_shadows()

        return total

    def refresh_shadows(self):
        """Refresh EgoRA shadow matrices."""
        if self._lora_modules:
            from egora.adapters import refresh_all_shadows
            refresh_all_shadows(self._lora_modules, momentum=self.config.shadow_momentum)

    def merge_and_unload(self):
        """Merge LoRA weights into base model and return the base model.

        Returns:
            The base model with LoRA weights merged in.
        """
        if self._lora_modules:
            from egora.adapters import merge_lora
            merge_lora(self.model, self._lora_modules)
        return self.model

    def get_nb_trainable_parameters(self) -> tuple:
        """Return (trainable_params, all_params) count.

        Returns:
            Tuple of (trainable, total) parameter counts.
        """
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def print_trainable_parameters(self):
        """Print trainable parameter summary."""
        trainable, total = self.get_nb_trainable_parameters()
        pct = trainable / total * 100 if total > 0 else 0
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")

    def save_pretrained(self, output_dir: str):
        """Save EgoRA config and adapter weights.

        Args:
            output_dir: Directory to save to.
        """
        os.makedirs(output_dir, exist_ok=True)

        self.config.save(os.path.join(output_dir, "egora_config.json"))

        adapter_state = {}
        for i, m in enumerate(self._lora_modules):
            adapter_state[f"adapter.{i}.lora_A"] = m.lora_A.data.cpu()
            adapter_state[f"adapter.{i}.lora_B"] = m.lora_B.data.cpu()
            if hasattr(m, "W_shadow"):
                adapter_state[f"adapter.{i}.W_shadow"] = m.W_shadow.cpu()

        torch.save(adapter_state, os.path.join(output_dir, "adapter_model.bin"))

        gov_state = self.governor.state_dict()
        torch.save(gov_state, os.path.join(output_dir, "governor_state.bin"))

    def load_adapter(self, input_dir: str):
        """Load adapter weights from a saved directory.

        Args:
            input_dir: Directory to load from.
        """
        adapter_path = os.path.join(input_dir, "adapter_model.bin")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"No adapter weights found at {adapter_path}")

        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        for i, m in enumerate(self._lora_modules):
            key_a = f"adapter.{i}.lora_A"
            key_b = f"adapter.{i}.lora_B"
            if key_a in state:
                m.lora_A.data.copy_(state[key_a])
            if key_b in state:
                m.lora_B.data.copy_(state[key_b])
            key_shadow = f"adapter.{i}.W_shadow"
            if key_shadow in state and hasattr(m, "W_shadow"):
                m.W_shadow.copy_(state[key_shadow])

        gov_path = os.path.join(input_dir, "governor_state.bin")
        if os.path.exists(gov_path):
            gov_state = torch.load(gov_path, map_location="cpu", weights_only=True)
            self.governor.load_state_dict(gov_state)
