"""Unit tests for the plotting module."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from egora.geometry import compute_head_geometry


class TinyTransformer(nn.Module):
    def __init__(self, d=32, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Module()
            layer.q_proj = nn.Linear(d, d, bias=False)
            layer.k_proj = nn.Linear(d, d, bias=False)
            layer.v_proj = nn.Linear(d, d, bias=False)
            layer.o_proj = nn.Linear(d, d, bias=False)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.o_proj(layer.v_proj(x))
        return x


def _make_results():
    torch.manual_seed(42)
    base = TinyTransformer()
    torch.manual_seed(42)
    tuned = TinyTransformer()
    with torch.no_grad():
        for p in tuned.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return compute_head_geometry(base, tuned)


@pytest.fixture
def results():
    return _make_results()


def _skip_if_no_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        pytest.skip("matplotlib not installed")


def test_plot_rotation_heatmap(results):
    _skip_if_no_matplotlib()
    from egora.plotting import plot_rotation_heatmap
    fig = plot_rotation_heatmap(results)
    assert fig is not None
    assert len(fig.axes) >= 1


def test_plot_layer_profile(results):
    _skip_if_no_matplotlib()
    from egora.plotting import plot_layer_profile
    fig = plot_layer_profile(results)
    assert fig is not None


def test_plot_mode_distribution(results):
    _skip_if_no_matplotlib()
    from egora.plotting import plot_mode_distribution
    fig = plot_mode_distribution(results)
    assert fig is not None


def test_plot_projection_comparison(results):
    _skip_if_no_matplotlib()
    from egora.plotting import plot_projection_comparison
    fig = plot_projection_comparison(results)
    assert fig is not None


def test_plot_rotation_report(results):
    _skip_if_no_matplotlib()
    from egora.plotting import plot_rotation_report
    fig = plot_rotation_report(results)
    assert fig is not None
    assert len(fig.axes) >= 4  # 4 panels


def test_plot_report_with_custom_title(results):
    _skip_if_no_matplotlib()
    from egora.plotting import plot_rotation_report
    fig = plot_rotation_report(results, suptitle="Test Report")
    assert fig is not None


def test_plot_heatmap_empty_data():
    _skip_if_no_matplotlib()
    from egora.plotting import plot_rotation_heatmap
    with pytest.raises(ValueError, match="No head data"):
        plot_rotation_heatmap({"heads": []})


def test_plot_save_to_file(results, tmp_path):
    _skip_if_no_matplotlib()
    from egora.plotting import plot_rotation_report
    fig = plot_rotation_report(results)
    path = str(tmp_path / "test_report.png")
    fig.savefig(path, dpi=72)
    import os
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1000  # non-trivial image
