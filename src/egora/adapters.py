"""
EgoRA Adapter Layers
====================
Drop-in LoRA adapter implementations with optional EgoRA entropy-governed
orthogonality regularization.

Adapter types:
  - EgoRALoRALinear  — Standard LoRA + optional EgoRA entropy penalty
  - DoRALinear       — Weight-Decomposed LoRA (Liu et al., 2024)
  - rsLoRALinear     — Rank-Stabilized LoRA (Kalajdzievski, 2023)

Usage::

    from egora import apply_lora, get_total_egora_penalty, refresh_all_shadows

    lora_modules = apply_lora(model, rank=16, use_egora=True)

    # In training loop:
    penalty = get_total_egora_penalty(lora_modules)
    loss = task_loss + lambda_coeff * penalty

    # Periodically:
    refresh_all_shadows(lora_modules, momentum=0.9)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EgoRALoRALinear(nn.Module):
    """LoRA adapter with optional EgoRA penalty on the low-rank product A·B.

    Standard LoRA: output = base_linear(x) + (x @ A^T) @ B^T × (α/r)
    EgoRA addition:  ||A·B · pinv(A·B)^T - I|| penalty (spectral regularization)

    The EgoRA penalty encourages the effective LoRA update (A·B) to have
    well-conditioned singular values, preventing rank collapse and
    promoting diverse adaptation directions.
    """

    def __init__(self, base_linear, rank=16, lora_alpha=32, use_egora=False,
                 lora_dropout=0.05):
        super().__init__()
        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.scaling = lora_alpha / rank
        self.use_egora = use_egora

        # Freeze base weights
        for p in self.base_linear.parameters():
            p.requires_grad = False

        # LoRA matrices: A (in→rank), B (rank→out) — always float32
        _dev = base_linear.weight.device
        self.lora_A = nn.Parameter(
            torch.randn(rank, self.in_features, device=_dev, dtype=torch.float32)
            * (1.0 / self.in_features ** 0.5)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=_dev, dtype=torch.float32)
        )
        self.lora_dropout = nn.Dropout(lora_dropout)

        # EgoRA shadow buffer for the effective weight W_lora = B @ A
        if use_egora:
            with torch.no_grad():
                W_eff = self.lora_B @ self.lora_A  # (out, in)
                shadow = self._compute_shadow(W_eff)
            self.register_buffer('W_shadow', shadow)

    @staticmethod
    def _compute_shadow(w):
        """Compute pseudo-inverse shadow: pinv(W)^T, same shape as W."""
        try:
            return torch.linalg.pinv(w).t()
        except Exception:
            out_f, in_f = w.shape
            if out_f <= in_f:
                return torch.linalg.solve(
                    w @ w.t() + 1e-4 * torch.eye(out_f, device=w.device), w)
            else:
                return torch.linalg.solve(
                    w.t() @ w + 1e-4 * torch.eye(in_f, device=w.device), w.t()).t()

    def forward(self, x):
        base_out = self.base_linear(x)
        x_f32 = x.float()
        lora_out = self.lora_dropout(x_f32) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        return base_out + lora_out.to(base_out.dtype)

    def get_raw_symmetry_loss(self):
        """||W_lora @ shadow^T - I||_F — measures spectral conditioning."""
        if not self.use_egora:
            return torch.tensor(0.0, device=self.lora_A.device)
        b_norm = self.lora_B.data.norm().item()
        if b_norm < 1e-6:
            return torch.tensor(0.0, device=self.lora_A.device, requires_grad=False)
        W_eff = self.lora_B @ self.lora_A  # (out, in)
        if self.out_features <= self.in_features:
            P = W_eff @ self.W_shadow.t()  # (out, out)
            I = torch.eye(self.out_features, device=W_eff.device)
        else:
            P = self.W_shadow.t() @ W_eff  # (in, in)
            I = torch.eye(self.in_features, device=W_eff.device)
        diff = P - I
        fro_sq = (diff * diff).sum()
        return torch.sqrt(fro_sq + 1e-8)

    def refresh_shadow(self, momentum=0.0):
        """Refresh shadow from current LoRA weights."""
        if not self.use_egora:
            return
        with torch.no_grad():
            W_eff = self.lora_B @ self.lora_A
            new_shadow = self._compute_shadow(W_eff)
            if momentum > 0:
                self.W_shadow.copy_(momentum * self.W_shadow + (1 - momentum) * new_shadow)
            else:
                self.W_shadow.copy_(new_shadow)


class DoRALinear(nn.Module):
    """DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024).

    Decomposes pretrained weight W into magnitude m and direction V/||V||.
    Trains LoRA on direction component only, with learnable magnitude vector.
    W' = m * (W + B@A) / ||W + B@A||_col
    """

    def __init__(self, base_linear, rank=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.scaling = lora_alpha / rank
        self.use_egora = False

        for p in self.base_linear.parameters():
            p.requires_grad = False

        _dev = base_linear.weight.device
        self.lora_A = nn.Parameter(
            torch.randn(rank, self.in_features, device=_dev, dtype=torch.float32)
            * (1.0 / self.in_features ** 0.5)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=_dev, dtype=torch.float32)
        )
        self.lora_dropout = nn.Dropout(lora_dropout)

        with torch.no_grad():
            W = base_linear.weight.float()
            self.magnitude = nn.Parameter(W.norm(dim=0).to(_dev))

    def forward(self, x):
        W = self.base_linear.weight.float()
        delta = self.scaling * (self.lora_B @ self.lora_A)
        W_adapted = W + delta
        col_norms = W_adapted.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_dora = self.magnitude.unsqueeze(0) * (W_adapted / col_norms)
        x_f32 = self.lora_dropout(x.float())
        out = F.linear(x_f32, W_dora.to(x_f32.dtype))
        if self.base_linear.bias is not None:
            out = out + self.base_linear.bias.float()
        return out.to(x.dtype)

    def get_raw_symmetry_loss(self):
        return torch.tensor(0.0, device=self.lora_A.device)

    def refresh_shadow(self, momentum=0.0):
        pass


class rsLoRALinear(nn.Module):
    """rsLoRA: Rank-Stabilized Low-Rank Adaptation (Kalajdzievski, 2023).

    Uses scaling factor alpha/sqrt(r) instead of alpha/r.
    """

    def __init__(self, base_linear, rank=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.scaling = lora_alpha / math.sqrt(rank)
        self.use_egora = False

        for p in self.base_linear.parameters():
            p.requires_grad = False

        _dev = base_linear.weight.device
        self.lora_A = nn.Parameter(
            torch.randn(rank, self.in_features, device=_dev, dtype=torch.float32)
            * (1.0 / self.in_features ** 0.5)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=_dev, dtype=torch.float32)
        )
        self.lora_dropout = nn.Dropout(lora_dropout)

    def forward(self, x):
        base_out = self.base_linear(x)
        x_f32 = self.lora_dropout(x.float())
        lora_out = x_f32 @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        return base_out + lora_out.to(base_out.dtype)

    def get_raw_symmetry_loss(self):
        return torch.tensor(0.0, device=self.lora_A.device)

    def refresh_shadow(self, momentum=0.0):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════════

ADAPTER_CLASSES = (EgoRALoRALinear, DoRALinear, rsLoRALinear)


def apply_lora(model, rank=16, lora_alpha=32, use_egora=False,
               target_modules=None, lora_dropout=0.05, adapter_type='egora'):
    """Replace attention projections with LoRA adapters.

    Args:
        model: A PyTorch model (e.g., a HuggingFace transformer).
        rank: LoRA rank.
        lora_alpha: LoRA scaling numerator.
        use_egora: Enable EgoRA entropy-governed penalty (only for adapter_type='egora').
        target_modules: List of submodule attribute names to replace.
            Default: ``['q_proj', 'k_proj', 'v_proj', 'o_proj']``.
        lora_dropout: Dropout probability on the LoRA path.
        adapter_type: One of ``'egora'``, ``'dora'``, ``'rslora'``.

    Returns:
        List of adapter modules (for penalty collection / shadow refresh).
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    # Freeze ALL parameters first
    for p in model.parameters():
        p.requires_grad = False

    lora_modules = []
    replaced = 0

    for name, module in model.named_modules():
        for attr_name in target_modules:
            if hasattr(module, attr_name):
                old_linear = getattr(module, attr_name)
                if not isinstance(old_linear, nn.Linear):
                    continue
                if adapter_type == 'dora':
                    new_module = DoRALinear(
                        old_linear, rank=rank, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout
                    )
                elif adapter_type == 'rslora':
                    new_module = rsLoRALinear(
                        old_linear, rank=rank, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout
                    )
                else:
                    new_module = EgoRALoRALinear(
                        old_linear, rank=rank, lora_alpha=lora_alpha,
                        use_egora=use_egora, lora_dropout=lora_dropout
                    )
                setattr(module, attr_name, new_module)
                lora_modules.append(new_module)
                replaced += 1

    return lora_modules


def get_total_egora_penalty(lora_modules):
    """Sum raw symmetry loss across all EgoRA-LoRA modules.

    Args:
        lora_modules: List of adapter modules returned by :func:`apply_lora`.

    Returns:
        Scalar tensor with the total penalty (differentiable).
    """
    device = lora_modules[0].lora_A.device if lora_modules else 'cpu'
    total = torch.tensor(0.0, device=device)
    for m in lora_modules:
        total = total + m.get_raw_symmetry_loss()
    return total


def refresh_all_shadows(lora_modules, momentum=0.0):
    """Refresh shadows for all EgoRA-LoRA modules.

    Args:
        lora_modules: List of adapter modules returned by :func:`apply_lora`.
        momentum: EMA momentum for shadow update (0 = hard replace).
    """
    for m in lora_modules:
        m.refresh_shadow(momentum)


def merge_lora(model, lora_modules):
    """Merge LoRA weights into base model weights (in-place).

    After merging, the model can be used for inference without the adapter
    wrappers. This is a destructive operation — the LoRA deltas are baked
    into the base weights.

    Args:
        model: The model containing the adapters.
        lora_modules: List of adapter modules returned by :func:`apply_lora`.

    Returns:
        List of ``(parent_module, attr_name, adapter)`` tuples that were
        replaced. Use :func:`unmerge_lora` to restore them.
    """
    replacements = []
    with torch.no_grad():
        for m in lora_modules:
            if isinstance(m, DoRALinear):
                W = m.base_linear.weight.float()
                delta = m.scaling * (m.lora_B @ m.lora_A)
                W_adapted = W + delta
                col_norms = W_adapted.norm(dim=0, keepdim=True).clamp(min=1e-8)
                W_merged = m.magnitude.unsqueeze(0) * (W_adapted / col_norms)
                m.base_linear.weight.data.copy_(W_merged.to(m.base_linear.weight.dtype))
            else:
                delta = m.scaling * (m.lora_B @ m.lora_A)
                m.base_linear.weight.data += delta.to(m.base_linear.weight.dtype)

    for name, parent_module in model.named_modules():
        for attr_name, child in list(parent_module._modules.items()):
            if isinstance(child, ADAPTER_CLASSES):
                replacements.append((parent_module, attr_name, child))
                setattr(parent_module, attr_name, child.base_linear)

    return replacements


def unmerge_lora(replacements):
    """Restore LoRA adapter wrappers after a :func:`merge_lora` call.

    Args:
        replacements: The list returned by :func:`merge_lora`.
    """
    for parent_module, attr_name, lora_child in replacements:
        setattr(parent_module, attr_name, lora_child)
