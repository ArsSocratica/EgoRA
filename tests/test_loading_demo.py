"""Tests for adapter loading and demo helpers (no network)."""

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from egora.adapters import apply_lora  # noqa: E402
from egora.geometry import compute_head_geometry  # noqa: E402
from egora.loading import detect_reference_kind, load_tuned_model  # noqa: E402
from egora.peft_compat import EgoRALoraConfig  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    config = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
    )
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(config).eval()


def _save_egora_adapter(base, path):
    model = copy.deepcopy(base)
    config = EgoRALoraConfig(r=4, lora_alpha=8)
    modules = apply_lora(model, rank=4, lora_alpha=8, use_egora=False)
    state = {}
    torch.manual_seed(2)
    for i, m in enumerate(modules):
        with torch.no_grad():
            m.lora_B.normal_(std=0.05)
        state[f"adapter.{i}.lora_A"] = m.lora_A.data.clone()
        state[f"adapter.{i}.lora_B"] = m.lora_B.data.clone()
    path.mkdir(exist_ok=True)
    config.save(str(path / "egora_config.json"))
    torch.save(state, str(path / "adapter_model.bin"))


def test_detect_and_merge_egora_adapter(tiny_model, tmp_path):
    adapter_dir = tmp_path / "adapter"
    _save_egora_adapter(tiny_model, adapter_dir)
    assert detect_reference_kind(str(adapter_dir)) == "egora_adapter"

    tuned, kind = load_tuned_model(tiny_model, str(adapter_dir))
    assert kind == "egora_adapter"
    # base model untouched, tuned model rotated
    same = compute_head_geometry(tiny_model, tiny_model)["theta_bar_deg"]
    assert same == pytest.approx(0.0, abs=0.05)
    assert compute_head_geometry(tiny_model, tuned)["theta_bar_deg"] > 0.01


def test_detect_peft_adapter(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    assert detect_reference_kind(str(tmp_path)) == "peft_adapter"


def test_demo_tables(tiny_model):
    pytest.importorskip("pandas")
    from egora.demo import format_report_markdown, layer_table, top_units_table

    tuned = copy.deepcopy(tiny_model)
    with torch.no_grad():
        for p in tuned.parameters():
            p.add_(0.01 * torch.randn_like(p))
    res = compute_head_geometry(tiny_model, tuned, granularity="head")
    md = format_report_markdown(res)
    assert "Heads analyzed" in md and "θ_crit" in md
    assert len(layer_table(res)) == 2
    assert len(top_units_table(res, n=5)) == 5


def test_build_demo():
    pytest.importorskip("gradio")
    from egora.demo import build_demo

    assert build_demo() is not None
