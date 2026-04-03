"""
EgoRA Entropy Governor
=======================
The core dynamic scaling mechanism that computes the entropy-driven
penalty coefficient lambda during training.

The governor computes:
  lambda = alpha * H(p)   (with optional floor, adaptive alpha, etc.)

where H(p) is the Shannon entropy of the model's output distribution.

Usage::

    from egora.governor import EntropyGovernor

    gov = EntropyGovernor(alpha=1.359, lam_floor=0.6931)

    # In training loop:
    lambda_coeff = gov.compute_lambda(logits)
    loss = task_loss + lambda_coeff * egora_penalty

    # At eval checkpoints (for adaptive alpha):
    gov.update_alpha(train_loss, val_loss)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class GovernorConfig:
    """Configuration for the entropy governor.

    Attributes:
        alpha: Base scaling coefficient. Default ``e/2 ≈ 1.359``.
        lam_floor: Minimum lambda value (prevents penalty collapse).
        adaptive_alpha: Enable adaptive alpha controller.
        alpha_min: Lower bound for adaptive alpha.
        alpha_max: Upper bound for adaptive alpha.
        alpha_gamma: Learning rate for alpha updates.
        gap_ema_beta: EMA decay for generalization gap smoothing.
        alpha_warmup: Number of eval steps before alpha adaptation begins.
        gap_spike_thresh: Relative gap change threshold for emergency boost.
        alpha_cooldown: Enable cooldown decay when gap stabilizes.
        alpha_cooldown_decay: Decay factor during cooldown.
        alpha_cooldown_patience: Consecutive stable evals before cooldown.
    """
    alpha: float = 1.359  # e/2
    lam_floor: float = 0.6931  # ln(2)
    adaptive_alpha: bool = False
    alpha_min: float = 0.05
    alpha_max: float = 3.0
    alpha_gamma: float = 0.15
    gap_ema_beta: float = 0.7
    alpha_warmup: int = 2
    gap_spike_thresh: float = 0.5
    alpha_cooldown: bool = False
    alpha_cooldown_decay: float = 0.95
    alpha_cooldown_patience: int = 3


class EntropyGovernor:
    """Dynamic entropy-driven scaling coefficient controller.

    Computes ``lambda = max(alpha * H(p), lam_floor)`` where ``H(p)`` is the
    Shannon entropy of the model's output probability distribution.

    Optionally adapts ``alpha`` based on the generalization gap trend.

    Args:
        config: A :class:`GovernorConfig` instance, or ``None`` for defaults.
        **kwargs: Override any :class:`GovernorConfig` field.
    """

    def __init__(self, config: Optional[GovernorConfig] = None, **kwargs):
        if config is None:
            config = GovernorConfig(**kwargs)
        self.config = config
        self.alpha = config.alpha
        self._alpha_baseline = config.alpha

        # Adaptive state
        self._gap_ema: Optional[float] = None
        self._prev_gap: Optional[float] = None
        self._eval_count: int = 0
        self._gap_stable_count: int = 0

    def compute_lambda(self, logits: torch.Tensor) -> float:
        """Compute the dynamic scaling coefficient from model logits.

        Args:
            logits: Raw model output logits, shape ``(B, T, V)`` or ``(B, V)``.

        Returns:
            The scalar lambda coefficient (float).
        """
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(-1).mean()
            ent_val = entropy.item()
            if math.isnan(ent_val) or math.isinf(ent_val):
                ent_val = 1.0
            info_lambda = self.alpha * ent_val
            info_lambda = max(info_lambda, self.config.lam_floor)
        return info_lambda

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute Shannon entropy of the output distribution.

        Args:
            logits: Raw model output logits.

        Returns:
            Scalar entropy value (float).
        """
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(-1).mean()
            return entropy.item()

    def update_alpha(self, train_loss: float, val_loss: float) -> float:
        """Update alpha based on the generalization gap trend.

        Should be called at each evaluation checkpoint during training.
        Only active when ``config.adaptive_alpha`` is True.

        Args:
            train_loss: Current training loss.
            val_loss: Current validation loss.

        Returns:
            Updated alpha value.
        """
        if not self.config.adaptive_alpha:
            return self.alpha

        cfg = self.config
        gap = val_loss - train_loss
        self._eval_count += 1

        if self._gap_ema is None:
            self._gap_ema = gap
        else:
            self._gap_ema = cfg.gap_ema_beta * self._gap_ema + (1 - cfg.gap_ema_beta) * gap

        if self._prev_gap is not None and self._eval_count > cfg.alpha_warmup:
            gap_trend = self._gap_ema - self._prev_gap
            rel_trend = gap_trend / (abs(self._prev_gap) + 1e-6)
            rel_trend = max(-1.0, min(1.0, rel_trend))

            if rel_trend > cfg.gap_spike_thresh:
                self.alpha = self.alpha * (1 + cfg.alpha_gamma * 3.0)
                self._gap_stable_count = 0
            else:
                self.alpha = self.alpha * (1 + cfg.alpha_gamma * rel_trend)

            self.alpha = max(cfg.alpha_min, min(cfg.alpha_max, self.alpha))

            # Cooldown
            if cfg.alpha_cooldown:
                if abs(rel_trend) < 0.05:
                    self._gap_stable_count += 1
                else:
                    self._gap_stable_count = 0
                if (self._gap_stable_count >= cfg.alpha_cooldown_patience
                        and self.alpha > self._alpha_baseline * 1.1):
                    self.alpha = (self._alpha_baseline
                                  + (self.alpha - self._alpha_baseline) * cfg.alpha_cooldown_decay)
                    self.alpha = max(cfg.alpha_min, self.alpha)

        self._prev_gap = self._gap_ema
        return self.alpha

    def state_dict(self) -> dict:
        """Serialize governor state for checkpointing."""
        return {
            'alpha': self.alpha,
            'alpha_baseline': self._alpha_baseline,
            'gap_ema': self._gap_ema,
            'prev_gap': self._prev_gap,
            'eval_count': self._eval_count,
            'gap_stable_count': self._gap_stable_count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore governor state from a checkpoint."""
        self.alpha = state['alpha']
        self._alpha_baseline = state['alpha_baseline']
        self._gap_ema = state.get('gap_ema')
        self._prev_gap = state.get('prev_gap')
        self._eval_count = state.get('eval_count', 0)
        self._gap_stable_count = state.get('gap_stable_count', 0)
