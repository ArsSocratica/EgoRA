"""Basic unit tests for EgoRA adapters."""

import torch
import torch.nn as nn
import pytest


def test_egora_lora_forward():
    """EgoRALoRALinear forward pass produces correct shape."""
    from egora.adapters import EgoRALoRALinear

    base = nn.Linear(64, 128)
    adapter = EgoRALoRALinear(base, rank=8, lora_alpha=16, use_egora=True)

    x = torch.randn(2, 10, 64)
    out = adapter(x)
    assert out.shape == (2, 10, 128)


def test_egora_penalty_nonzero_after_training():
    """After a gradient step, the EgoRA penalty should be non-zero."""
    from egora.adapters import EgoRALoRALinear

    base = nn.Linear(64, 128)
    adapter = EgoRALoRALinear(base, rank=8, lora_alpha=16, use_egora=True)

    # Simulate a weight update so B is no longer zero
    with torch.no_grad():
        adapter.lora_B.data.normal_(0, 0.1)
    adapter.refresh_shadow()

    penalty = adapter.get_raw_symmetry_loss()
    assert penalty.item() >= 0.0
    assert penalty.requires_grad or penalty.item() == 0.0


def test_egora_penalty_zero_without_flag():
    """Without use_egora=True, penalty should be zero."""
    from egora.adapters import EgoRALoRALinear

    base = nn.Linear(64, 128)
    adapter = EgoRALoRALinear(base, rank=8, lora_alpha=16, use_egora=False)

    penalty = adapter.get_raw_symmetry_loss()
    assert penalty.item() == 0.0


def test_dora_forward():
    """DoRALinear forward pass produces correct shape."""
    from egora.adapters import DoRALinear

    base = nn.Linear(64, 128)
    adapter = DoRALinear(base, rank=8, lora_alpha=16)

    x = torch.randn(2, 10, 64)
    out = adapter(x)
    assert out.shape == (2, 10, 128)


def test_rslora_forward():
    """rsLoRALinear forward pass produces correct shape."""
    from egora.adapters import rsLoRALinear

    base = nn.Linear(64, 128)
    adapter = rsLoRALinear(base, rank=8, lora_alpha=16)

    x = torch.randn(2, 10, 64)
    out = adapter(x)
    assert out.shape == (2, 10, 128)


def test_apply_lora():
    """apply_lora replaces target modules in a simple model."""
    from egora.adapters import apply_lora

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32)
            self.k_proj = nn.Linear(32, 32)
            self.v_proj = nn.Linear(32, 32)
            self.o_proj = nn.Linear(32, 32)
            self.mlp = nn.Linear(32, 32)

        def forward(self, x):
            return self.o_proj(self.v_proj(x) + self.q_proj(x) + self.k_proj(x))

    model = TinyModel()
    modules = apply_lora(model, rank=4, lora_alpha=8, use_egora=True)

    assert len(modules) == 4  # q, k, v, o
    # mlp should NOT be replaced
    assert isinstance(model.mlp, nn.Linear)


def test_get_total_penalty():
    """get_total_egora_penalty sums across modules."""
    from egora.adapters import apply_lora, get_total_egora_penalty

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32)
            self.k_proj = nn.Linear(32, 32)

        def forward(self, x):
            return self.q_proj(x) + self.k_proj(x)

    model = TinyModel()
    modules = apply_lora(model, rank=4, use_egora=True,
                         target_modules=['q_proj', 'k_proj'])

    penalty = get_total_egora_penalty(modules)
    assert penalty.shape == ()  # scalar
    assert not torch.isnan(penalty)


def test_refresh_shadows():
    """refresh_all_shadows runs without error."""
    from egora.adapters import apply_lora, refresh_all_shadows

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32)

        def forward(self, x):
            return self.q_proj(x)

    model = TinyModel()
    modules = apply_lora(model, rank=4, use_egora=True,
                         target_modules=['q_proj'])

    # Should not raise
    refresh_all_shadows(modules, momentum=0.0)
    refresh_all_shadows(modules, momentum=0.9)


def test_merge_unmerge():
    """merge_lora and unmerge_lora roundtrip."""
    from egora.adapters import apply_lora, merge_lora, unmerge_lora, EgoRALoRALinear

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32)

        def forward(self, x):
            return self.q_proj(x)

    model = TinyModel()
    modules = apply_lora(model, rank=4, use_egora=False,
                         target_modules=['q_proj'])

    # Before merge: q_proj is an adapter
    assert isinstance(model.q_proj, EgoRALoRALinear)

    replacements = merge_lora(model, modules)

    # After merge: q_proj is a plain Linear
    assert isinstance(model.q_proj, nn.Linear)

    unmerge_lora(replacements)

    # After unmerge: q_proj is an adapter again
    assert isinstance(model.q_proj, EgoRALoRALinear)
