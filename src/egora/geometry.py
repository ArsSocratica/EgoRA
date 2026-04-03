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
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F


def compute_head_geometry(
    base_model: torch.nn.Module,
    tuned_model: torch.nn.Module,
    target_modules: Optional[List[str]] = None,
    damage_threshold_deg: float = 5.0,
) -> Dict[str, Any]:
    """Compute per-head representational geometry between base and tuned model.

    Args:
        base_model: Pre-trained model (reference).
        tuned_model: Fine-tuned model (or model with merged LoRA).
        target_modules: Projection names to analyze.
            Default: ``['q_proj', 'k_proj', 'v_proj', 'o_proj']``.
        damage_threshold_deg: Rotation angle (degrees) above which a head
            is classified as ``"substitutive"`` or ``"damaged"``.

    Returns:
        Dict with per-head metrics and aggregate statistics.
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

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

        # Extract layer number
        layer_num = None
        for part in name.split('.'):
            if part.isdigit():
                layer_num = int(part)
                break

        with torch.no_grad():
            w_base = base_param.float().flatten()
            w_tuned = tuned_param.float().flatten()

            cos_sim = F.cosine_similarity(
                w_base.unsqueeze(0), w_tuned.unsqueeze(0)
            ).item()
            cos_sim = max(-1.0, min(1.0, cos_sim))

            theta_rad = np.arccos(cos_sim)
            theta_deg = float(np.degrees(theta_rad))

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

        head_info = {
            'name': name,
            'layer': layer_num,
            'proj_type': proj_type,
            'cosine_similarity': cos_sim,
            'rotation_angle_deg': theta_deg,
            'magnitude_ratio': mag_ratio,
            'learning_mode': mode,
        }
        heads.append(head_info)

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
    theta_std = float(np.std(all_angles))
    theta_max = float(np.max(all_angles))
    theta_min = float(np.min(all_angles))

    mode_counts = {
        m: all_modes.count(m)
        for m in ['preserved', 'additive', 'substitutive', 'damaged']
    }
    mode_fractions = {m: c / n_heads for m, c in mode_counts.items()}

    layer_means = {
        l: float(np.mean(angles))
        for l, angles in sorted(layer_rotations.items())
    }
    proj_means = {
        p: float(np.mean(angles))
        for p, angles in proj_rotations.items()
    }

    return {
        "n_heads": n_heads,
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
