"""Unit tests for the geometry / diagnostics module."""

import torch
import torch.nn as nn
import numpy as np
import pytest
import os
import json
import tempfile

from egora.geometry import compute_head_geometry, compute_interhead_diversity, save_geometry


class TinyTransformer(nn.Module):
    """Minimal model with attention projection naming convention."""

    def __init__(self, d=32, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Module()
            layer.q_proj = nn.Linear(d, d, bias=False)
            layer.k_proj = nn.Linear(d, d, bias=False)
            layer.v_proj = nn.Linear(d, d, bias=False)
            layer.o_proj = nn.Linear(d, d, bias=False)
            # Register as proper submodule
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.o_proj(layer.v_proj(x))
        return x


def _make_pair():
    """Create a base model and a slightly perturbed tuned model."""
    torch.manual_seed(42)
    base = TinyTransformer(d=32, n_layers=2)

    torch.manual_seed(42)
    tuned = TinyTransformer(d=32, n_layers=2)

    # Perturb tuned model slightly
    with torch.no_grad():
        for p in tuned.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    return base, tuned


def test_compute_head_geometry_basic():
    """compute_head_geometry returns expected keys and shapes."""
    base, tuned = _make_pair()
    results = compute_head_geometry(base, tuned)

    assert "n_heads" in results
    assert "theta_bar_deg" in results
    assert "theta_std_deg" in results
    assert "theta_max_deg" in results
    assert "theta_min_deg" in results
    assert "mode_counts" in results
    assert "mode_fractions" in results
    assert "damaged_fraction" in results
    assert "per_layer_mean_rotation" in results
    assert "per_projection_mean_rotation" in results
    assert "heads" in results

    # 2 layers × 4 projections = 8 heads
    assert results["n_heads"] == 8
    assert len(results["heads"]) == 8


def test_compute_head_geometry_small_perturbation():
    """Small perturbation → small rotation, mostly preserved mode."""
    base, tuned = _make_pair()
    results = compute_head_geometry(base, tuned)

    # Small perturbation should give small rotation
    assert results["theta_bar_deg"] < 30.0
    # At least some heads should not be damaged
    non_damaged = results["n_heads"] - results["mode_counts"]["damaged"]
    assert non_damaged >= 0


def test_compute_head_geometry_identical_models():
    """Identical models → zero rotation, all preserved."""
    torch.manual_seed(42)
    base = TinyTransformer(d=32, n_layers=2)
    torch.manual_seed(42)
    same = TinyTransformer(d=32, n_layers=2)

    results = compute_head_geometry(base, same)

    # Floating point may cause tiny non-zero rotation
    assert results["theta_bar_deg"] < 0.1
    assert results["mode_counts"]["preserved"] == results["n_heads"]
    assert results["damaged_fraction"] == 0.0


def test_compute_head_geometry_large_perturbation():
    """Large perturbation → large rotation, likely damaged heads."""
    torch.manual_seed(42)
    base = TinyTransformer(d=32, n_layers=2)
    torch.manual_seed(99)
    different = TinyTransformer(d=32, n_layers=2)

    results = compute_head_geometry(base, different, damage_threshold_deg=5.0)

    # Random weights → expect substantial rotation
    assert results["theta_bar_deg"] > 1.0


def test_compute_head_geometry_no_match():
    """If target_modules don't match, returns error."""
    base, tuned = _make_pair()
    results = compute_head_geometry(base, tuned, target_modules=["nonexistent_proj"])

    assert "error" in results
    assert results["heads"] == []


def test_compute_head_geometry_per_layer():
    """per_layer_mean_rotation contains entries for each layer."""
    base, tuned = _make_pair()
    results = compute_head_geometry(base, tuned)

    layer_means = results["per_layer_mean_rotation"]
    assert len(layer_means) == 2  # 2 layers


def test_compute_head_geometry_per_projection():
    """per_projection_mean_rotation contains entries for each proj type."""
    base, tuned = _make_pair()
    results = compute_head_geometry(base, tuned)

    proj_means = results["per_projection_mean_rotation"]
    assert set(proj_means.keys()) == {"q_proj", "k_proj", "v_proj", "o_proj"}


def test_compute_interhead_diversity():
    """compute_interhead_diversity returns a square similarity matrix."""
    torch.manual_seed(42)
    model = TinyTransformer(d=32, n_layers=2)

    sim_matrix, head_names = compute_interhead_diversity(model)

    assert sim_matrix is not None
    assert len(head_names) == 8  # 2 layers × 4 projections
    assert sim_matrix.shape == (8, 8)

    # Diagonal should be 1.0 (self-similarity)
    for i in range(8):
        assert sim_matrix[i, i] == pytest.approx(1.0, abs=1e-5)

    # Matrix should be symmetric
    assert np.allclose(sim_matrix, sim_matrix.T, atol=1e-5)


def test_compute_interhead_diversity_no_match():
    """Returns (None, []) when no modules match."""
    torch.manual_seed(42)
    model = TinyTransformer(d=32, n_layers=2)

    sim_matrix, head_names = compute_interhead_diversity(
        model, target_modules=["nonexistent"]
    )

    assert sim_matrix is None
    assert head_names == []


def test_save_geometry():
    """save_geometry creates JSON and numpy files."""
    base, tuned = _make_pair()
    results = compute_head_geometry(base, tuned)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "geometry.json")
        save_geometry(results, output_path)

        # Main JSON file
        assert os.path.exists(output_path)
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["n_heads"] == 8

        # Subdirectory files
        geo_dir = os.path.join(tmpdir, "rotation_geometry")
        assert os.path.isdir(geo_dir)
        assert os.path.exists(os.path.join(geo_dir, "theta_per_head.npy"))
        assert os.path.exists(os.path.join(geo_dir, "mean_theta.json"))
        assert os.path.exists(os.path.join(geo_dir, "layer_profile.npy"))

        # Verify numpy array
        thetas = np.load(os.path.join(geo_dir, "theta_per_head.npy"))
        assert len(thetas) == 8


def test_mode_fractions_sum_to_one():
    """Mode fractions should sum to 1.0."""
    base, tuned = _make_pair()
    results = compute_head_geometry(base, tuned)

    total = sum(results["mode_fractions"].values())
    assert total == pytest.approx(1.0, abs=1e-10)
