"""Unit tests for the EntropyGovernor."""

import math
import torch
import pytest

from egora.governor import EntropyGovernor, GovernorConfig


def test_compute_lambda_basic():
    """Governor returns a positive lambda from random logits."""
    gov = EntropyGovernor(alpha=1.359, lam_floor=0.6931)
    logits = torch.randn(2, 10, 100)  # (B, T, V)
    lam = gov.compute_lambda(logits)
    assert isinstance(lam, float)
    assert lam >= 0.6931


def test_lam_floor_enforced():
    """Lambda never drops below lam_floor, even with near-zero entropy."""
    gov = EntropyGovernor(alpha=0.01, lam_floor=0.5)
    # One-hot logits → near-zero entropy
    logits = torch.full((1, 1, 50), -100.0)
    logits[0, 0, 0] = 100.0
    lam = gov.compute_lambda(logits)
    assert lam >= 0.5


def test_high_entropy_high_lambda():
    """Uniform distribution → high entropy → lambda > floor."""
    gov = EntropyGovernor(alpha=1.359, lam_floor=0.01)
    # Uniform logits → max entropy
    logits = torch.zeros(1, 1, 1000)
    lam = gov.compute_lambda(logits)
    # H(uniform over 1000) ≈ 6.9, so lambda ≈ 1.359 * 6.9 ≈ 9.4
    assert lam > 5.0


def test_compute_entropy():
    """compute_entropy returns a sensible value."""
    gov = EntropyGovernor()
    logits = torch.randn(2, 5, 100)
    ent = gov.compute_entropy(logits)
    assert isinstance(ent, float)
    assert ent > 0


def test_adaptive_alpha_disabled():
    """update_alpha is a no-op when adaptive_alpha=False."""
    gov = EntropyGovernor(alpha=1.0)
    assert gov.config.adaptive_alpha is False
    old_alpha = gov.alpha
    gov.update_alpha(train_loss=1.0, val_loss=1.5)
    assert gov.alpha == old_alpha


def test_adaptive_alpha_changes():
    """With adaptive_alpha=True, alpha changes after warmup evals."""
    gov = EntropyGovernor(GovernorConfig(
        alpha=1.0,
        adaptive_alpha=True,
        alpha_warmup=1,
        alpha_gamma=0.15,
    ))
    # Feed several eval updates with increasing gap
    gov.update_alpha(1.0, 1.2)  # eval 1 (warmup)
    gov.update_alpha(1.0, 1.4)  # eval 2 (past warmup, gap growing)
    gov.update_alpha(1.0, 1.8)  # eval 3 (gap still growing → alpha should increase)
    assert gov.alpha != 1.0  # alpha has been adapted


def test_state_dict_roundtrip():
    """state_dict / load_state_dict preserves governor state."""
    gov = EntropyGovernor(GovernorConfig(alpha=2.0, adaptive_alpha=True))
    gov.update_alpha(1.0, 1.3)
    gov.update_alpha(1.0, 1.5)

    state = gov.state_dict()

    gov2 = EntropyGovernor(GovernorConfig(alpha=999.0))
    gov2.load_state_dict(state)

    assert gov2.alpha == gov.alpha
    assert gov2._gap_ema == gov._gap_ema
    assert gov2._eval_count == gov._eval_count
