"""Unit tests for the PEFT compatibility module."""

import os
import json
import pytest
import torch
import torch.nn as nn

from egora.peft_compat import EgoRALoraConfig, EgoRAPeftModel


class TinyModel(nn.Module):
    """Minimal model with attention projection naming convention."""

    def __init__(self, d=32):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.mlp = nn.Linear(d, d)

    def forward(self, x):
        return self.o_proj(self.v_proj(x) + self.q_proj(x) + self.k_proj(x))


# ── EgoRALoraConfig tests ────────────────────────────────────────────────────

def test_config_defaults():
    """Default config has sensible values."""
    config = EgoRALoraConfig()
    assert config.r == 16
    assert config.use_egora is True
    assert config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert config.egora_alpha == pytest.approx(1.359)


def test_config_custom():
    """Custom config overrides defaults."""
    config = EgoRALoraConfig(r=8, lora_alpha=16, use_egora=False)
    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.use_egora is False


def test_config_to_dict():
    """to_dict returns a serializable dict."""
    config = EgoRALoraConfig(r=32)
    d = config.to_dict()
    assert isinstance(d, dict)
    assert d["r"] == 32
    assert "target_modules" in d


def test_config_save_load(tmp_path):
    """Config roundtrips through save/load."""
    config = EgoRALoraConfig(r=64, egora_alpha=2.0)
    path = str(tmp_path / "config.json")
    config.save(path)

    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["r"] == 64

    loaded = EgoRALoraConfig.load(path)
    assert loaded.r == 64
    assert loaded.egora_alpha == 2.0


# ── EgoRAPeftModel tests ─────────────────────────────────────────────────────

def test_peft_model_wraps_model():
    """EgoRAPeftModel wraps a model and applies adapters."""
    model = TinyModel()
    config = EgoRALoraConfig(r=4, lora_alpha=8, use_egora=True)
    wrapped = EgoRAPeftModel(model, config)

    assert len(wrapped._lora_modules) == 4  # q, k, v, o
    assert wrapped.governor is not None


def test_peft_model_forward():
    """Forward pass works through the wrapper."""
    model = TinyModel()
    config = EgoRALoraConfig(r=4, use_egora=True)
    wrapped = EgoRAPeftModel(model, config)

    x = torch.randn(2, 10, 32)
    out = wrapped(x)
    assert out.shape == (2, 10, 32)


def test_peft_model_penalty():
    """compute_egora_penalty returns a scalar tensor."""
    model = TinyModel()
    config = EgoRALoraConfig(r=4, use_egora=True)
    wrapped = EgoRAPeftModel(model, config)

    penalty = wrapped.compute_egora_penalty()
    assert penalty.shape == ()
    assert not torch.isnan(penalty)


def test_peft_model_compute_lambda():
    """compute_lambda returns a float."""
    model = TinyModel()
    config = EgoRALoraConfig(r=4)
    wrapped = EgoRAPeftModel(model, config)

    logits = torch.randn(2, 10, 100)
    lam = wrapped.compute_lambda(logits)
    assert isinstance(lam, float)
    assert lam > 0


def test_peft_model_total_loss():
    """compute_total_loss returns task_loss + penalty."""
    model = TinyModel()
    config = EgoRALoraConfig(r=4, use_egora=True)
    wrapped = EgoRAPeftModel(model, config)

    task_loss = torch.tensor(2.0, requires_grad=True)
    logits = torch.randn(2, 10, 100)
    total = wrapped.compute_total_loss(task_loss, logits)
    assert total.item() >= task_loss.item()


def test_peft_model_trainable_params():
    """get_nb_trainable_parameters returns correct counts."""
    model = TinyModel()
    config = EgoRALoraConfig(r=4, use_egora=True)
    wrapped = EgoRAPeftModel(model, config)

    trainable, total = wrapped.get_nb_trainable_parameters()
    assert trainable > 0
    assert total > trainable  # base params are frozen


def test_peft_model_save_load(tmp_path):
    """save_pretrained / load_adapter roundtrip."""
    model = TinyModel()
    config = EgoRALoraConfig(r=4, use_egora=True)
    wrapped = EgoRAPeftModel(model, config)

    # Modify adapter weights
    with torch.no_grad():
        wrapped._lora_modules[0].lora_B.data.fill_(0.42)

    output_dir = str(tmp_path / "saved")
    wrapped.save_pretrained(output_dir)

    assert os.path.exists(os.path.join(output_dir, "egora_config.json"))
    assert os.path.exists(os.path.join(output_dir, "adapter_model.bin"))
    assert os.path.exists(os.path.join(output_dir, "governor_state.bin"))

    # Load into fresh model
    model2 = TinyModel()
    wrapped2 = EgoRAPeftModel(model2, config)
    wrapped2.load_adapter(output_dir)

    # Check weights match
    assert torch.allclose(
        wrapped2._lora_modules[0].lora_B.data,
        torch.full_like(wrapped2._lora_modules[0].lora_B.data, 0.42),
    )


def test_peft_model_merge():
    """merge_and_unload returns the base model."""
    model = TinyModel()
    config = EgoRALoraConfig(r=4, use_egora=False)
    wrapped = EgoRAPeftModel(model, config)

    base = wrapped.merge_and_unload()
    assert isinstance(base.q_proj, nn.Linear)  # merged back to plain Linear


def test_peft_model_no_egora():
    """Without use_egora, penalty is zero."""
    model = TinyModel()
    config = EgoRALoraConfig(r=4, use_egora=False)
    wrapped = EgoRAPeftModel(model, config)

    # Should still have adapters but no penalty
    assert len(wrapped._lora_modules) == 4
    penalty = wrapped.compute_egora_penalty()
    assert penalty.item() == 0.0
