"""Tests for architecture-aware model info, per-head geometry and verdicts."""

import copy
import math

import pytest
import torch

from egora._compat import normalize_model_kwargs, trainer_tokenizer_kwargs
from egora.adapters import apply_lora, effective_weight, merge_lora
from egora.geometry import compute_adapter_rotation, compute_head_geometry, rotation_verdict
from egora.model_info import get_attention_dims, summarize_model, theta_crit_deg

transformers = pytest.importorskip("transformers")

HIDDEN, N_HEADS, N_KV, HEAD_DIM, LAYERS = 64, 4, 2, 16, 2


@pytest.fixture(scope="module")
def tiny_gqa_model():
    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(config).eval()


# ── θ_crit / model info ──────────────────────────────────────────────────────


def test_theta_crit_values():
    assert theta_crit_deg(64) == pytest.approx(7.18, abs=0.01)
    assert theta_crit_deg(128) == pytest.approx(5.07, abs=0.01)
    assert theta_crit_deg(1) == pytest.approx(90.0)
    with pytest.raises(ValueError):
        theta_crit_deg(0)


def test_attention_dims_from_config(tiny_gqa_model):
    dims = get_attention_dims(tiny_gqa_model)
    assert dims["head_dim"] == HEAD_DIM
    assert dims["num_attention_heads"] == N_HEADS
    assert dims["num_key_value_heads"] == N_KV


def test_summary_is_gqa_aware(tiny_gqa_model):
    info = summarize_model(tiny_gqa_model)
    assert info["head_dim"] == HEAD_DIM
    assert info["theta_crit_deg"] == pytest.approx(math.degrees(math.asin(1 / 4)))
    kv_out = N_KV * HEAD_DIM
    per_layer_in_out = (HIDDEN + HIDDEN) * 2 + (HIDDEN + kv_out) * 2  # q,o + k,v
    for r in (8, 16):
        assert info["lora_estimates"][r]["params"] == r * per_layer_in_out * LAYERS
    assert info["target_count"] == 4 * LAYERS


def test_trainable_params_match_estimate(tiny_gqa_model):
    model = copy.deepcopy(tiny_gqa_model)
    est = summarize_model(model)["lora_estimates"][8]["params"]
    apply_lora(model, rank=8, use_egora=False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == est


# ── Geometry ─────────────────────────────────────────────────────────────────


def test_head_granularity_counts_and_localizes(tiny_gqa_model):
    tuned = copy.deepcopy(tiny_gqa_model)
    with torch.no_grad():
        q = tuned.model.layers[1].self_attn.q_proj.weight
        q[2 * HEAD_DIM:3 * HEAD_DIM] += 0.5 * torch.randn(HEAD_DIM, HIDDEN)

    proj = compute_head_geometry(tiny_gqa_model, tuned)
    assert proj["granularity"] == "projection"
    assert proj["n_heads"] == 4 * LAYERS
    assert proj["head_dim"] == HEAD_DIM

    heads = compute_head_geometry(tiny_gqa_model, tuned, granularity="head")
    assert heads["n_heads"] == LAYERS * (N_HEADS + N_KV + N_KV + N_HEADS)
    rotated = [h for h in heads["heads"] if h["rotation_angle_deg"] > 1.0]
    assert len(rotated) == 1
    assert (rotated[0]["layer"], rotated[0]["proj_type"], rotated[0]["head"]) == (1, "q_proj", 2)
    # A single rotated head is diluted at projection level.
    proj_q = [h for h in proj["heads"] if h["layer"] == 1 and h["proj_type"] == "q_proj"][0]
    assert rotated[0]["rotation_angle_deg"] > proj_q["rotation_angle_deg"]


def test_output_projection_split_by_columns(tiny_gqa_model):
    tuned = copy.deepcopy(tiny_gqa_model)
    with torch.no_grad():
        o = tuned.model.layers[0].self_attn.o_proj.weight
        o[:, HEAD_DIM:2 * HEAD_DIM] += 0.5 * torch.randn(HIDDEN, HEAD_DIM)
    heads = compute_head_geometry(tiny_gqa_model, tuned, granularity="head")
    rotated = [h for h in heads["heads"] if h["rotation_angle_deg"] > 1.0]
    assert [(h["layer"], h["proj_type"], h["head"]) for h in rotated] == [(0, "o_proj", 1)]


def test_head_granularity_requires_head_dim():
    a = torch.nn.Sequential(torch.nn.Linear(8, 8))
    with pytest.raises(ValueError):
        compute_head_geometry(a, copy.deepcopy(a), target_modules=["0"], granularity="head")


def test_default_verdict_uses_model_theta_crit(tiny_gqa_model):
    res = compute_head_geometry(tiny_gqa_model, copy.deepcopy(tiny_gqa_model))
    assert res["verdict"]["theta_crit_deg"] == pytest.approx(theta_crit_deg(HEAD_DIM))
    assert res["verdict"]["level"] == "safe"
    fixed = compute_head_geometry(
        tiny_gqa_model, copy.deepcopy(tiny_gqa_model), verdict_theta_crit=5.0
    )
    assert fixed["verdict"]["theta_crit_deg"] == 5.0


@pytest.mark.parametrize(
    "theta,crit,level",
    [(2.0, None, "safe"), (4.0, None, "warning"), (6.0, None, "danger"),
     (4.0, 7.18, "safe"), (5.0, 7.18, "warning"), (7.5, 7.18, "danger")],
)
def test_rotation_verdict_levels(theta, crit, level):
    v = rotation_verdict(theta, crit)
    assert v["level"] == level
    if crit is None:
        assert (v["theta_crit_deg"], v["warning_deg"]) == (5.0, 3.0)


# ── Adapters ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("adapter_type", ["egora", "dora", "rslora"])
def test_adapter_rotation_matches_merged_geometry(tiny_gqa_model, adapter_type):
    model = copy.deepcopy(tiny_gqa_model)
    modules = apply_lora(model, rank=4, use_egora=False, adapter_type=adapter_type)
    torch.manual_seed(1)
    with torch.no_grad():
        for m in modules:
            m.lora_B.normal_(std=0.05)
    live = compute_adapter_rotation(modules)
    assert live["theta_bar_deg"] > 0
    assert len(live["per_module_deg"]) == 4 * LAYERS

    w_before = [effective_weight(m).clone() for m in modules]
    merge_lora(model, modules)
    for m, w in zip(modules, w_before):
        assert torch.allclose(m.base_linear.weight.float(), w, atol=1e-5)
    merged = compute_head_geometry(tiny_gqa_model, model)
    assert merged["theta_bar_deg"] == pytest.approx(live["theta_bar_deg"], abs=1e-3)


def test_adapter_rotation_empty():
    assert compute_adapter_rotation([])["theta_bar_deg"] == 0.0


# ── Compat ───────────────────────────────────────────────────────────────────


def test_normalize_model_kwargs_renames_dtype():
    out = normalize_model_kwargs({"torch_dtype": torch.float16, "x": 1})
    assert out["x"] == 1
    assert torch.float16 in out.values()
    assert len(out) == 2


def test_trainer_tokenizer_kwargs():
    class New:
        def __init__(self, processing_class=None):
            pass

    class Old:
        def __init__(self, tokenizer=None):
            pass

    assert trainer_tokenizer_kwargs(New, "tok") == {"processing_class": "tok"}
    assert trainer_tokenizer_kwargs(Old, "tok") == {"tokenizer": "tok"}
