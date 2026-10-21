"""
Representational Geometry Analysis
===================================
Computes per-head rotation metrics between base and fine-tuned models.

Metrics per head:
  - cosine_similarity: cos(theta) between base and tuned weight vectors
  - rotation_angle_deg: theta in degrees
  - magnitude_ratio: ||w_tuned|| / ||w_base||
  - learning_mode: additive / substitutive / preserved / damaged

Aggregate metrics:
  - theta_bar (mean rotation angle)
  - damaged_fraction
  - per-layer rotation gradient
  - per-projection-type breakdown (Q, K, V, O)

Usage::

    from egora.geometry import compute_head_geometry, compute_interhead_diversity

    results = compute_head_geometry(base_model, tuned_model)
    print(f"Mean rotation: {results['theta_bar_deg']:.2f} degrees")
    print(f"Damaged heads: {results['damaged_fraction']*100:.1f}%")
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

VERDICT_FIXED_CRIT_DEG = 5.0
VERDICT_WARNING_RATIO = 0.6

_INPUT_SPLIT_PROJECTIONS = ("o_proj", "out_proj", "dense")


def rotation_verdict(
    theta_bar_deg: float,
    theta_crit: Optional[float] = None,
) -> Dict[str, Any]:
    """Classify a mean rotation against the Rotation-Retention thresholds.

    Args:
        theta_bar_deg: Mean rotation angle θ̄ in degrees.
        theta_crit: Critical angle in degrees. Defaults to the fixed 5°.
            Pass the model-specific ``arcsin(1/√d_head)`` to scale thresholds
            to the architecture.

    Returns:
        Dict with ``level`` (``"safe"``, ``"warning"`` or ``"danger"``),
        ``message``, ``theta_crit_deg`` and ``warning_deg``
        (``0.6 × θ_crit``, i.e. 3° for the fixed 5°).
    """
    crit = float(theta_crit) if theta_crit else VERDICT_FIXED_CRIT_DEG
    warn = VERDICT_WARNING_RATIO * crit
    if theta_bar_deg < warn:
        level, msg = "safe", f"SAFE — Knowledge well preserved (θ̄ < {warn:.2f}°)"
    elif theta_bar_deg < crit:
        level, msg = "warning", f"WARNING — Moderate rotation ({warn:.2f}° ≤ θ̄ < {crit:.2f}°)"
    else:
        level, msg = "danger", f"DANGER — Significant knowledge loss expected (θ̄ ≥ {crit:.2f}°)"
    return {"level": level, "message": msg, "theta_crit_deg": crit, "warning_deg": warn}


def _unit_metrics(w_base: torch.Tensor, w_tuned: torch.Tensor, damage_threshold_deg: float):
    w_base = w_base.float().flatten()
    w_tuned = w_tuned.float().flatten()

    cos_sim = F.cosine_similarity(w_base.unsqueeze(0), w_tuned.unsqueeze(0)).item()
    cos_sim = max(-1.0, min(1.0, cos_sim))
    theta_deg = float(np.degrees(np.arccos(cos_sim)))

    mag_base = w_base.norm().item()
    mag_tuned = w_tuned.norm().item()
    mag_ratio = mag_tuned / max(mag_base, 1e-10)

    # Learning mode classification (patent Section 14E)
    if theta_deg < damage_threshold_deg and 0.9 < mag_ratio < 1.1:
        mode = "preserved"
    elif theta_deg < damage_threshold_deg and mag_ratio >= 1.1:
        mode = "additive"
    elif theta_deg >= damage_threshold_deg and mag_ratio >= 0.9:
        mode = "substitutive"
    else:
        mode = "damaged"
    return cos_sim, theta_deg, mag_ratio, mode


def _split_heads(w: torch.Tensor, proj_type: str, head_dim: int):
    """Split a projection weight into per-head blocks.

    Q/K/V weights are ``(n_heads * d_head, hidden)`` and are split along rows;
    output projections are ``(hidden, n_heads * d_head)`` and are split along
    columns. Returns ``None`` if the shape is not divisible by ``head_dim``.
    """
    dim = 1 if proj_type in _INPUT_SPLIT_PROJECTIONS else 0
    if w.dim() != 2 or w.shape[dim] % head_dim != 0:
        return None
    return list(torch.split(w, head_dim, dim=dim))


def compute_head_geometry(
    base_model: torch.nn.Module,
    tuned_model: torch.nn.Module,
    target_modules: Optional[List[str]] = None,
    damage_threshold_deg: float = 5.0,
    granularity: str = "projection",
    head_dim: Optional[int] = None,
    verdict_theta_crit: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute representational rotation geometry between base and tuned model.

    Args:
        base_model: Pre-trained model (reference).
        tuned_model: Fine-tuned model (or model with merged LoRA).
        target_modules: Projection names to analyze.
            Default: ``['q_proj', 'k_proj', 'v_proj', 'o_proj']``.
        damage_threshold_deg: Rotation angle (degrees) above which a unit
            is classified as ``"substitutive"`` or ``"damaged"``.
        granularity: ``"projection"`` (default) measures each whole
            projection matrix; ``"head"`` splits every projection into
            attention heads (GQA-aware: K/V use the key/value head count).
        head_dim: Attention head dimension. Read from ``base_model.config``
            when omitted. Required for ``granularity="head"``.
        verdict_theta_crit: Critical angle for the ``verdict``. Defaults to the
            model-specific ``arcsin(1/√d_head)`` when ``head_dim`` is known,
            else 5°. Pass ``5.0`` for the fixed thresholds.

    Returns:
        Dict with per-unit metrics (``heads``) and aggregate statistics.
        ``n_heads`` is the number of analyzed units (projections or heads,
        see ``granularity``). ``theta_crit_deg`` is set when ``head_dim``
        is known.
    """
    from egora.model_info import get_attention_dims, theta_crit_deg

    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    if granularity not in ("projection", "head"):
        raise ValueError(f"granularity must be 'projection' or 'head', got {granularity!r}")

    if head_dim is None:
        head_dim = get_attention_dims(base_model)["head_dim"]
    if granularity == "head" and not head_dim:
        raise ValueError(
            "granularity='head' needs head_dim: pass it explicitly or use a model "
            "with a HuggingFace config"
        )

    heads: List[Dict[str, Any]] = []
    layer_rotations: Dict[int, List[float]] = defaultdict(list)
    proj_rotations: Dict[str, List[float]] = defaultdict(list)

    base_params = dict(base_model.named_parameters())
    tuned_params = dict(tuned_model.named_parameters())

    for name, base_param in base_params.items():
        proj_type = None
        for tm in target_modules:
            if f".{tm}." in name or name.endswith(f".{tm}.weight"):
                proj_type = tm
                break
        if proj_type is None:
            continue
        if not name.endswith('.weight'):
            continue
        if name not in tuned_params:
            continue

        tuned_param = tuned_params[name]

        layer_num = None
        for part in name.split('.'):
            if part.isdigit():
                layer_num = int(part)
                break

        with torch.no_grad():
            w_base = base_param.detach().cpu()
            w_tuned = tuned_param.detach().cpu()
            units = [(None, w_base, w_tuned)]
            if granularity == "head":
                base_blocks = _split_heads(w_base, proj_type, head_dim)
                tuned_blocks = _split_heads(w_tuned, proj_type, head_dim)
                if base_blocks is not None and tuned_blocks is not None:
                    units = [
                        (i, b, t) for i, (b, t) in enumerate(zip(base_blocks, tuned_blocks))
                    ]

            for head_idx, wb, wt in units:
                cos_sim, theta_deg, mag_ratio, mode = _unit_metrics(
                    wb, wt, damage_threshold_deg
                )
                heads.append({
                    'name': name if head_idx is None else f"{name}[head={head_idx}]",
                    'layer': layer_num,
                    'proj_type': proj_type,
                    'head': head_idx,
                    'cosine_similarity': cos_sim,
                    'rotation_angle_deg': theta_deg,
                    'magnitude_ratio': mag_ratio,
                    'learning_mode': mode,
                })
                if layer_num is not None:
                    layer_rotations[layer_num].append(theta_deg)
                proj_rotations[proj_type].append(theta_deg)

    # Aggregate statistics
    all_angles = [h['rotation_angle_deg'] for h in heads]
    all_modes = [h['learning_mode'] for h in heads]

    n_heads = len(heads)
    if n_heads == 0:
        return {"error": "No matching parameters found", "heads": []}

    theta_bar = float(np.mean(all_angles))
    theta_crit = theta_crit_deg(head_dim) if head_dim else None
    verdict = rotation_verdict(theta_bar, verdict_theta_crit or theta_crit)
    theta_std = float(np.std(all_angles))
    theta_max = float(np.max(all_angles))
    theta_min = float(np.min(all_angles))

    mode_counts = {
        m: all_modes.count(m)
        for m in ['preserved', 'additive', 'substitutive', 'damaged']
    }
    mode_fractions = {m: c / n_heads for m, c in mode_counts.items()}

    layer_means = {
        layer: float(np.mean(angles))
        for layer, angles in sorted(layer_rotations.items())
    }
    proj_means = {
        p: float(np.mean(angles))
        for p, angles in proj_rotations.items()
    }

    return {
        "n_heads": n_heads,
        "granularity": granularity,
        "head_dim": head_dim,
        "theta_crit_deg": theta_crit,
        "verdict": verdict,
        "damage_threshold_deg": damage_threshold_deg,
        "theta_bar_deg": theta_bar,
        "theta_std_deg": theta_std,
        "theta_max_deg": theta_max,
        "theta_min_deg": theta_min,
        "mode_counts": mode_counts,
        "mode_fractions": mode_fractions,
        "damaged_fraction": mode_fractions.get('damaged', 0.0),
        "per_layer_mean_rotation": layer_means,
        "per_projection_mean_rotation": proj_means,
        "heads": heads,
    }


def compute_interhead_diversity(
    model: torch.nn.Module,
    target_modules: Optional[List[str]] = None,
) -> tuple:
    """Compute cosine similarity matrix between all attention head weight vectors.

    Measures head diversity — EgoRA should preserve diversity.

    Args:
        model: A PyTorch model.
        target_modules: Projection names to include.

    Returns:
        Tuple of ``(sim_matrix, head_names)`` where ``sim_matrix`` is an
        NxN numpy array and ``head_names`` is a list of parameter names.
        Returns ``(None, [])`` if no matching parameters are found.
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    head_vectors = []
    head_names = []

    for name, param in model.named_parameters():
        proj_type = None
        for tm in target_modules:
            if f".{tm}." in name or name.endswith(f".{tm}.weight"):
                proj_type = tm
                break
        if proj_type is None or not name.endswith('.weight'):
            continue

        with torch.no_grad():
            head_vectors.append(param.float().flatten().cpu())
            head_names.append(name)

    if not head_vectors:
        return None, []

    N = len(head_vectors)
    sim_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            if head_vectors[i].shape == head_vectors[j].shape:
                cs = F.cosine_similarity(
                    head_vectors[i].unsqueeze(0),
                    head_vectors[j].unsqueeze(0),
                ).item()
            else:
                cs = 0.0
            sim_matrix[i, j] = cs
            sim_matrix[j, i] = cs

    return sim_matrix, head_names


def compute_adapter_rotation(lora_modules) -> Dict[str, Any]:
    """Rotation of each adapted projection relative to its frozen base weight.

    Cheap enough to call during training: no model copy or merge is needed.

    Args:
        lora_modules: Adapter modules returned by :func:`egora.apply_lora`.

    Returns:
        Dict with ``theta_bar_deg``, ``theta_max_deg`` and per-module angles
        (``per_module_deg``).
    """
    from egora.adapters import effective_weight

    angles = []
    with torch.no_grad():
        for m in lora_modules:
            w_base = m.base_linear.weight.float()
            w_eff = effective_weight(m)
            cos = F.cosine_similarity(w_base.flatten().unsqueeze(0), w_eff.flatten().unsqueeze(0))
            angles.append(float(np.degrees(np.arccos(max(-1.0, min(1.0, cos.item()))))))
    if not angles:
        return {"theta_bar_deg": 0.0, "theta_max_deg": 0.0, "per_module_deg": []}
    return {
        "theta_bar_deg": float(np.mean(angles)),
        "theta_max_deg": float(np.max(angles)),
        "per_module_deg": angles,
    }


def save_geometry(results: Dict[str, Any], output_path: str) -> None:
    """Save geometry results to JSON + numpy arrays.

    Creates a ``rotation_geometry/`` subfolder next to *output_path* with:
      - ``theta_per_head.npy``
      - ``mean_theta.json``
      - ``layer_profile.npy``
      - ``interhead_diversity.npy`` (if present)

    Args:
        results: Dict returned by :func:`compute_head_geometry`.
        output_path: Path for the main JSON output file.
    """
    geo_dir = os.path.join(os.path.dirname(output_path), 'rotation_geometry')
    os.makedirs(geo_dir, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(_convert(results), f, indent=2)

    if 'heads' in results:
        thetas = np.array([h['rotation_angle_deg'] for h in results['heads']])
        np.save(os.path.join(geo_dir, 'theta_per_head.npy'), thetas)

    mean_theta = {k: _convert(v) for k, v in results.items() if k != 'heads'}
    with open(os.path.join(geo_dir, 'mean_theta.json'), 'w') as f:
        json.dump(mean_theta, f, indent=2)

    if 'per_layer_mean_rotation' in results:
        layers = results['per_layer_mean_rotation']
        if layers:
            max_layer = max(int(k) for k in layers.keys())
            profile = np.zeros(max_layer + 1)
            for k, v in layers.items():
                profile[int(k)] = v
            np.save(os.path.join(geo_dir, 'layer_profile.npy'), profile)

    if 'interhead_diversity' in results and results['interhead_diversity'] is not None:
        np.save(
            os.path.join(geo_dir, 'interhead_diversity.npy'),
            np.array(results['interhead_diversity']),
        )
