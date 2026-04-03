"""
EgoRA Visualization
====================
Publication-quality plots for rotation geometry analysis.

Requires ``matplotlib`` (install with ``pip install egora[diagnostics]``).

Functions:
  - plot_rotation_heatmap: Per-layer × per-projection rotation heatmap
  - plot_layer_profile: Rotation angle across layers
  - plot_mode_distribution: Learning mode pie/bar chart
  - plot_rotation_report: Combined 4-panel report figure
"""

from typing import Dict, Any, Optional, List
import numpy as np


def _check_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install egora[diagnostics]"
        )


def plot_rotation_heatmap(
    results: Dict[str, Any],
    ax=None,
    cmap: str = "RdYlGn_r",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
):
    """Plot per-layer × per-projection rotation heatmap.

    Args:
        results: Dict returned by :func:`egora.compute_head_geometry`.
        ax: Matplotlib axes. Created if None.
        cmap: Colormap name.
        vmin: Minimum rotation angle for colorbar.
        vmax: Maximum rotation angle for colorbar (auto if None).

    Returns:
        Matplotlib Figure (only if ax was None).
    """
    plt = _check_matplotlib()

    heads = results.get("heads", [])
    if not heads:
        raise ValueError("No head data in results")

    # Build matrix: rows=layers, cols=projection types
    proj_types = sorted(set(h["proj_type"] for h in heads))
    layers = sorted(set(h["layer"] for h in heads if h["layer"] is not None))

    matrix = np.full((len(layers), len(proj_types)), np.nan)
    layer_idx = {l: i for i, l in enumerate(layers)}
    proj_idx = {p: i for i, p in enumerate(proj_types)}

    for h in heads:
        if h["layer"] is not None and h["proj_type"] in proj_idx:
            matrix[layer_idx[h["layer"]], proj_idx[h["proj_type"]]] = h["rotation_angle_deg"]

    if vmax is None:
        vmax = max(10.0, np.nanmax(matrix) * 1.1)

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3 + len(proj_types) * 0.8, 1 + len(layers) * 0.25))
    else:
        fig = ax.figure

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(len(proj_types)))
    ax.set_xticklabels(proj_types, fontsize=9)
    ax.set_xlabel("Projection Type")

    # Show subset of layer labels if many
    if len(layers) > 40:
        step = max(1, len(layers) // 20)
        ax.set_yticks(range(0, len(layers), step))
        ax.set_yticklabels([str(layers[i]) for i in range(0, len(layers), step)], fontsize=7)
    else:
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(l) for l in layers], fontsize=7)
    ax.set_ylabel("Layer")

    ax.set_title("Rotation Angle (degrees)", fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="θ (degrees)")

    # Draw critical threshold line in colorbar
    if vmax >= 5.0:
        cbar.ax.axhline(y=5.0, color="red", linewidth=1.5, linestyle="--", alpha=0.8)

    if created_fig:
        fig.tight_layout()
        return fig


def plot_layer_profile(
    results: Dict[str, Any],
    ax=None,
    show_threshold: bool = True,
):
    """Plot mean rotation angle per layer.

    Args:
        results: Dict returned by :func:`egora.compute_head_geometry`.
        ax: Matplotlib axes. Created if None.
        show_threshold: Draw the 5° critical threshold line.

    Returns:
        Matplotlib Figure (only if ax was None).
    """
    plt = _check_matplotlib()

    layer_means = results.get("per_layer_mean_rotation", {})
    if not layer_means:
        raise ValueError("No per-layer data in results")

    layers = sorted(layer_means.keys(), key=lambda x: int(x))
    angles = [layer_means[l] for l in layers]
    x = [int(l) for l in layers]

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure

    ax.fill_between(x, angles, alpha=0.3, color="#2196F3")
    ax.plot(x, angles, color="#1565C0", linewidth=1.5, marker=".", markersize=3)

    if show_threshold:
        ax.axhline(y=5.0, color="red", linewidth=1.0, linestyle="--", alpha=0.7, label="θ_crit = 5°")
        ax.axhline(y=3.0, color="orange", linewidth=1.0, linestyle=":", alpha=0.6, label="Warning = 3°")
        ax.legend(fontsize=8, loc="upper right")

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Mean θ (degrees)", fontsize=10)
    ax.set_title("Rotation Profile Across Layers", fontsize=11, fontweight="bold")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(bottom=0)

    if created_fig:
        fig.tight_layout()
        return fig


def plot_mode_distribution(
    results: Dict[str, Any],
    ax=None,
    colors: Optional[Dict[str, str]] = None,
):
    """Plot learning mode distribution as horizontal bar chart.

    Args:
        results: Dict returned by :func:`egora.compute_head_geometry`.
        ax: Matplotlib axes. Created if None.
        colors: Dict mapping mode names to colors.

    Returns:
        Matplotlib Figure (only if ax was None).
    """
    plt = _check_matplotlib()

    mode_counts = results.get("mode_counts", {})
    if not mode_counts:
        raise ValueError("No mode data in results")

    if colors is None:
        colors = {
            "preserved": "#4CAF50",
            "additive": "#2196F3",
            "substitutive": "#FF9800",
            "damaged": "#F44336",
        }

    modes = ["preserved", "additive", "substitutive", "damaged"]
    counts = [mode_counts.get(m, 0) for m in modes]
    fracs = results.get("mode_fractions", {})
    bar_colors = [colors.get(m, "#999999") for m in modes]

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 2.5))
    else:
        fig = ax.figure

    bars = ax.barh(modes, counts, color=bar_colors, edgecolor="white", height=0.7)

    for bar, mode in zip(bars, modes):
        frac = fracs.get(mode, 0)
        if frac > 0.02:
            ax.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{frac*100:.1f}%", va="center", fontsize=9, color="#333"
            )

    ax.set_xlabel("Number of Heads", fontsize=10)
    ax.set_title("Learning Mode Distribution", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    if created_fig:
        fig.tight_layout()
        return fig


def plot_projection_comparison(
    results: Dict[str, Any],
    ax=None,
):
    """Plot mean rotation per projection type as grouped bar chart.

    Args:
        results: Dict returned by :func:`egora.compute_head_geometry`.
        ax: Matplotlib axes. Created if None.

    Returns:
        Matplotlib Figure (only if ax was None).
    """
    plt = _check_matplotlib()

    proj_means = results.get("per_projection_mean_rotation", {})
    if not proj_means:
        raise ValueError("No per-projection data in results")

    proj_types = sorted(proj_means.keys())
    angles = [proj_means[p] for p in proj_types]

    proj_colors = {
        "q_proj": "#E91E63",
        "k_proj": "#9C27B0",
        "v_proj": "#3F51B5",
        "o_proj": "#009688",
    }
    colors = [proj_colors.get(p, "#607D8B") for p in proj_types]

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    else:
        fig = ax.figure

    bars = ax.bar(proj_types, angles, color=colors, edgecolor="white", width=0.6)

    for bar, angle in zip(bars, angles):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{angle:.1f}°", ha="center", va="bottom", fontsize=9
        )

    ax.axhline(y=5.0, color="red", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_ylabel("Mean θ (degrees)", fontsize=10)
    ax.set_title("Rotation by Projection Type", fontsize=11, fontweight="bold")
    ax.set_ylim(bottom=0)

    if created_fig:
        fig.tight_layout()
        return fig


def plot_rotation_report(
    results: Dict[str, Any],
    figsize: tuple = (14, 10),
    suptitle: Optional[str] = None,
):
    """Generate a combined 4-panel rotation geometry report.

    Panels:
      (A) Rotation heatmap (layer × projection)
      (B) Layer profile (rotation across depth)
      (C) Learning mode distribution
      (D) Per-projection comparison

    Args:
        results: Dict returned by :func:`egora.compute_head_geometry`.
        figsize: Figure size.
        suptitle: Custom title. Auto-generated if None.

    Returns:
        Matplotlib Figure.
    """
    plt = _check_matplotlib()
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    theta_bar = results.get("theta_bar_deg", 0)
    n_heads = results.get("n_heads", 0)

    if suptitle is None:
        suptitle = f"EgoRA Rotation Geometry Report — θ̄ = {theta_bar:.2f}° ({n_heads} heads)"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)

    # (A) Heatmap
    ax_heatmap = fig.add_subplot(gs[0, 0])
    try:
        plot_rotation_heatmap(results, ax=ax_heatmap)
    except ValueError:
        ax_heatmap.text(0.5, 0.5, "No heatmap data", ha="center", va="center")
    ax_heatmap.set_title("(A) Rotation Heatmap", fontsize=11, fontweight="bold")

    # (B) Layer profile
    ax_profile = fig.add_subplot(gs[0, 1])
    try:
        plot_layer_profile(results, ax=ax_profile)
    except ValueError:
        ax_profile.text(0.5, 0.5, "No layer data", ha="center", va="center")
    ax_profile.set_title("(B) Layer Profile", fontsize=11, fontweight="bold")

    # (C) Mode distribution
    ax_modes = fig.add_subplot(gs[1, 0])
    try:
        plot_mode_distribution(results, ax=ax_modes)
    except ValueError:
        ax_modes.text(0.5, 0.5, "No mode data", ha="center", va="center")
    ax_modes.set_title("(C) Learning Modes", fontsize=11, fontweight="bold")

    # (D) Projection comparison
    ax_proj = fig.add_subplot(gs[1, 1])
    try:
        plot_projection_comparison(results, ax=ax_proj)
    except ValueError:
        ax_proj.text(0.5, 0.5, "No projection data", ha="center", va="center")
    ax_proj.set_title("(D) Projection Comparison", fontsize=11, fontweight="bold")

    # Verdict banner
    if theta_bar < 3.0:
        verdict = "✅ SAFE — Knowledge preserved"
        color = "#4CAF50"
    elif theta_bar < 5.0:
        verdict = "⚠️ WARNING — Check benchmarks"
        color = "#FF9800"
    else:
        verdict = "❌ DANGER — Knowledge loss expected"
        color = "#F44336"

    fig.text(
        0.5, 0.01, verdict,
        ha="center", fontsize=12, fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9),
    )

    return fig
