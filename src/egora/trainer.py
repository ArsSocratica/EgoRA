"""
EgoRA Trainer — HuggingFace Trainer Integration
=================================================
Drop-in replacement for ``transformers.Trainer`` that automatically applies
EgoRA adapters, entropy-governed penalty, and shadow refresh.

Usage::

    from egora import EgoRATrainer, EgoRATrainingArguments, EgoRALoraConfig

    config = EgoRALoraConfig(r=16, lora_alpha=32, use_egora=True)
    args = EgoRATrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
    )

    trainer = EgoRATrainer(
        model_name="meta-llama/Llama-3.2-1B",
        egora_config=config,
        args=args,
        train_dataset=dataset,
    )
    trainer.train()

    # Post-training diagnostics
    report = trainer.diagnose()
    print(f"Mean rotation: {report['theta_bar_deg']:.2f}°")

Requires ``transformers>=4.30`` (imported lazily).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


@dataclass
class EgoRATrainingArguments:
    """Training arguments combining HuggingFace TrainingArguments with EgoRA defaults.

    This is a thin wrapper: all fields are forwarded to
    ``transformers.TrainingArguments``. The defaults here are tuned for
    LoRA-scale fine-tuning.

    Attributes:
        output_dir: Output directory for checkpoints and logs.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per device.
        per_device_eval_batch_size: Eval batch size per device.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of steps for LR warmup.
        lr_scheduler_type: LR scheduler type.
        logging_steps: Log every N steps.
        save_strategy: Checkpoint save strategy.
        eval_strategy: Evaluation strategy.
        bf16: Use bf16 mixed precision (CUDA). Auto-detected if None.
        fp16: Use fp16 mixed precision (MPS/older GPUs). Auto-detected if None.
        gradient_accumulation_steps: Gradient accumulation steps.
        max_grad_norm: Maximum gradient norm for clipping.
        report_to: Reporting integrations (e.g., "wandb", "tensorboard").
        extra_args: Any additional kwargs forwarded to TrainingArguments.
    """
    output_dir: str = "./egora-output"
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "no"
    bf16: Optional[bool] = None
    fp16: Optional[bool] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    report_to: Optional[Union[str, List[str]]] = "none"
    extra_args: Optional[Dict[str, Any]] = None

    def _resolve_bf16(self) -> bool:
        """Resolve bf16 setting: explicit > auto-detect (CUDA only)."""
        if self.bf16 is not None:
            return self.bf16
        return torch.cuda.is_available()

    def _resolve_fp16(self) -> bool:
        """Resolve fp16 setting: explicit > auto-detect.

        Note: fp16 is NOT auto-enabled on MPS because it causes NaN in
        softmax/log operations. MPS accelerates fp32 natively on Apple Silicon.
        """
        if self.fp16 is not None:
            return self.fp16
        return False

    def to_hf_args(self):
        """Convert to a ``transformers.TrainingArguments`` instance."""
        try:
            from transformers import TrainingArguments
        except ImportError:
            raise ImportError(
                "transformers is required for EgoRATrainer. "
                "Install with: pip install transformers"
            )
        kwargs = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "eval_strategy": self.eval_strategy,
            "bf16": self._resolve_bf16(),
            "fp16": self._resolve_fp16(),
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "report_to": self.report_to,
            "remove_unused_columns": False,
        }
        if self.extra_args:
            kwargs.update(self.extra_args)
        return TrainingArguments(**kwargs)


class EgoRATrainer:
    """High-level trainer that wraps HuggingFace Trainer with EgoRA.

    Handles model loading, adapter injection, entropy-governed training,
    shadow refresh, and post-training diagnostics.

    Args:
        model_name: HuggingFace model name or local path.
        egora_config: An :class:`~egora.peft_compat.EgoRALoraConfig`.
        args: An :class:`EgoRATrainingArguments` instance.
        train_dataset: Training dataset (HuggingFace Dataset or PyTorch Dataset).
        eval_dataset: Optional evaluation dataset.
        tokenizer: Optional tokenizer (auto-loaded from model_name if None).
        data_collator: Optional data collator.
        model: Optional pre-loaded model (skips loading from model_name).
        model_kwargs: Additional kwargs for model loading (e.g., torch_dtype).
        callbacks: Additional Trainer callbacks.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        egora_config=None,
        args: Optional[EgoRATrainingArguments] = None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        model=None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        callbacks: Optional[list] = None,
    ):
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainerCallback,
            )
        except ImportError:
            raise ImportError(
                "transformers>=4.30 is required for EgoRATrainer. "
                "Install with: pip install 'transformers>=4.30'"
            )

        from egora.peft_compat import EgoRALoraConfig
        from egora.adapters import apply_lora, get_total_egora_penalty, refresh_all_shadows
        from egora.governor import EntropyGovernor, GovernorConfig

        # Defaults
        if egora_config is None:
            egora_config = EgoRALoraConfig()
        if args is None:
            args = EgoRATrainingArguments()

        self.egora_config = egora_config
        self.training_args = args
        self._hf_args = args.to_hf_args()

        # Load model
        if model is not None:
            self.model = model
        else:
            if model_name is None:
                raise ValueError("Either model_name or model must be provided.")
            _mkwargs = model_kwargs or {}
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **_mkwargs)

        # Store base state for diagnostics
        self._base_state = {
            k: v.clone().cpu() for k, v in self.model.state_dict().items()
        }
        self._model_name = model_name

        # Load tokenizer
        if tokenizer is None and model_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        # Apply EgoRA adapters
        self._lora_modules = apply_lora(
            self.model,
            rank=egora_config.r,
            lora_alpha=egora_config.lora_alpha,
            use_egora=egora_config.use_egora,
            target_modules=egora_config.target_modules,
            lora_dropout=egora_config.lora_dropout,
            adapter_type=egora_config.adapter_type,
        )

        # Governor — pass all config parameters
        self._governor = EntropyGovernor(GovernorConfig(
            alpha=egora_config.egora_alpha,
            lam_floor=egora_config.egora_lam_floor,
            adaptive_alpha=egora_config.egora_adaptive_alpha,
            alpha_min=egora_config.egora_alpha_min,
            alpha_max=egora_config.egora_alpha_max,
            alpha_gamma=egora_config.egora_alpha_gamma,
            gap_ema_beta=egora_config.egora_gap_ema_beta,
            alpha_warmup=egora_config.egora_alpha_warmup,
            gap_spike_thresh=egora_config.egora_gap_spike_thresh,
            alpha_cooldown=egora_config.egora_alpha_cooldown,
            alpha_cooldown_decay=egora_config.egora_alpha_cooldown_decay,
            alpha_cooldown_patience=egora_config.egora_alpha_cooldown_patience,
        ))

        # Auto-enable evaluation when adaptive alpha needs it
        if egora_config.egora_adaptive_alpha and eval_dataset is not None:
            if self._hf_args.eval_strategy == "no":
                self._hf_args.eval_strategy = "epoch"
                logger.info(
                    "Adaptive alpha enabled with eval_dataset: "
                    "auto-set eval_strategy='epoch'"
                )

        # Step counter and loss tracking for shadow refresh + adaptive alpha
        self._step_count = 0
        self._last_train_loss: Optional[float] = None
        shadow_interval = egora_config.shadow_refresh_interval

        # Capture references for the closure
        lora_modules = self._lora_modules
        governor = self._governor
        trainer_self = self

        # Custom Trainer subclass with EgoRA loss override
        class _EgoRAHFTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                outputs = model(**inputs)
                task_loss = outputs.loss

                lam = governor.compute_lambda(outputs.logits)
                penalty = get_total_egora_penalty(lora_modules)
                total_loss = task_loss + lam * penalty

                # Track for adaptive alpha
                trainer_self._last_train_loss = task_loss.item()

                trainer_self._step_count += 1
                if (shadow_interval > 0
                        and trainer_self._step_count % shadow_interval == 0):
                    refresh_all_shadows(lora_modules, momentum=egora_config.shadow_momentum)

                if return_outputs:
                    return total_loss, outputs
                return total_loss

        # EgoRA callback: logging + adaptive alpha after evaluation
        class _EgoRACallback(TrainerCallback):
            def on_log(self, _args, state, control, logs=None, **kwargs):
                if logs is not None:
                    with torch.no_grad():
                        penalty_val = get_total_egora_penalty(lora_modules).item()
                    logs["egora_penalty"] = penalty_val
                    logs["egora_lambda"] = governor.alpha
                    logs["egora_alpha"] = governor.alpha
                    logs["egora_step"] = trainer_self._step_count

            def on_evaluate(self, _args, state, control, metrics=None, **kwargs):
                """After each evaluation, update adaptive alpha using the
                generalization gap (val_loss - train_loss)."""
                if not governor.config.adaptive_alpha:
                    return
                if metrics is None:
                    return
                val_loss = metrics.get("eval_loss")
                train_loss = trainer_self._last_train_loss
                if val_loss is None or train_loss is None:
                    return
                old_alpha = governor.alpha
                new_alpha = governor.update_alpha(train_loss, val_loss)
                gap = val_loss - train_loss
                logger.info(
                    f"Adaptive alpha update: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"gap={gap:.4f}, alpha: {old_alpha:.4f} → {new_alpha:.4f}"
                )

        _callbacks = [_EgoRACallback()]
        if callbacks:
            _callbacks.extend(callbacks)

        # Build the HF Trainer
        self._trainer = _EgoRAHFTrainer(
            model=self.model,
            args=self._hf_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=_callbacks,
        )

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"EgoRATrainer ready: {len(self._lora_modules)} adapters, "
            f"rank={egora_config.r}, "
            f"trainable={n_trainable:,}/{n_total:,} "
            f"({n_trainable/n_total*100:.2f}%)"
        )

    def train(self, **kwargs):
        """Start training. All kwargs forwarded to ``Trainer.train()``."""
        return self._trainer.train(**kwargs)

    def evaluate(self, **kwargs):
        """Run evaluation. All kwargs forwarded to ``Trainer.evaluate()``."""
        return self._trainer.evaluate(**kwargs)

    def save_model(self, output_dir: Optional[str] = None):
        """Save adapter weights and EgoRA config.

        Args:
            output_dir: Directory to save to (default: training output_dir).
        """
        output_dir = output_dir or self._hf_args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.egora_config.save(os.path.join(output_dir, "egora_config.json"))

        adapter_state = {}
        for i, m in enumerate(self._lora_modules):
            adapter_state[f"adapter.{i}.lora_A"] = m.lora_A.data.cpu()
            adapter_state[f"adapter.{i}.lora_B"] = m.lora_B.data.cpu()
            if hasattr(m, "W_shadow"):
                adapter_state[f"adapter.{i}.W_shadow"] = m.W_shadow.cpu()
        torch.save(adapter_state, os.path.join(output_dir, "adapter_model.bin"))

        gov_state = self._governor.state_dict()
        torch.save(gov_state, os.path.join(output_dir, "governor_state.bin"))
        logger.info(f"EgoRA model saved to {output_dir}")

    def merge_and_unload(self):
        """Merge LoRA weights into base model and return it.

        Returns:
            The base model with adapter weights merged in.
        """
        from egora.adapters import merge_lora
        merge_lora(self.model, self._lora_modules)
        return self.model

    def diagnose(self) -> dict:
        """Run post-training rotation diagnostics.

        Compares current model weights against the stored base state
        to compute per-head rotation geometry.

        Returns:
            Geometry results dict with keys like ``theta_bar_deg``,
            ``damaged_fraction``, ``per_head``, etc.
        """
        from egora.geometry import compute_head_geometry

        # Build a lightweight base reference from stored state
        import copy
        base_model = copy.deepcopy(self.model)

        # Merge adapters in the current model first
        from egora.adapters import merge_lora, unmerge_lora
        replacements = merge_lora(self.model, self._lora_modules)

        try:
            # Load original base weights into the copy
            base_model.load_state_dict(self._base_state, strict=False)
            results = compute_head_geometry(base_model, self.model)
        finally:
            # Restore adapters
            unmerge_lora(replacements)

        return results

    def print_trainable_parameters(self):
        """Print trainable parameter summary."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        pct = trainable / total * 100 if total > 0 else 0
        print(
            f"trainable params: {trainable:,} || "
            f"all params: {total:,} || "
            f"trainable%: {pct:.4f}"
        )
