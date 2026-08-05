"""Tensor kernels for Context-Conditioned Rational Activations.

The public layers live in :mod:`cca_core`; this module contains only the hot
pointwise path.  Keeping the functions small and free of Python-side data
dependent control flow lets ``torch.compile`` fuse each curve into one Triton
kernel on CUDA (and optimized native code on CPU).

Unlike the original implementation, coefficients stay in group form.  They
are broadcast over features/tokens/pixels instead of being materialized with
``repeat_interleave``.  This is both faster in eager mode and substantially
reduces the tensors saved for backward.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _rational_curve(
    x: Tensor,
    rat: Tensor,
    w: Tensor,
    d: Tensor,
    alpha: Tensor,
    coefficient_dim: int,
) -> Tensor:
    """Apply the complete CCA curve to already-broadcast group tensors."""
    a0, a1, a2, b0, b1 = rat.unbind(dim=coefficient_dim)
    u = w * x + d
    u2 = u * u
    numerator = a0 + a1 * u + a2 * u2
    denominator_base = b0 + b1 * u
    correction = alpha * numerator / (1.0 + denominator_base * denominator_base)
    return torch.nn.functional.silu(u) + correction


def _align_1d(tensor: Tensor, x_ndim: int) -> Tensor:
    """Align ``[*context, groups]`` with ``[*x, groups, group_width]``."""
    for _ in range(x_ndim - tensor.ndim):
        tensor = tensor.unsqueeze(-2)
    return tensor.unsqueeze(-1)


def cca_1d_kernel(
    x: Tensor,
    rat: Tensor,
    w: Tensor,
    d: Tensor,
    alpha: Tensor,
    groups: int,
) -> Tensor:
    """Group-native CCA kernel for ``[B,F]`` and ``[B,S,F]`` inputs.

    ``rat`` has shape ``[*context, 5, G]``. Context can be per-example or
    per-token; missing token axes are inserted as broadcast dimensions.
    """
    grouped = x.unflatten(-1, (groups, x.shape[-1] // groups))
    aligned_rat = torch.stack(
        [_align_1d(coefficient, x.ndim) for coefficient in rat.unbind(dim=-2)],
        dim=-2,
    )
    aligned_w = _align_1d(w.squeeze(-2), x.ndim)
    aligned_d = _align_1d(d.squeeze(-2), x.ndim)
    alpha_shape = (1,) * (x.ndim - 1) + (groups, 1)
    out = _rational_curve(
        grouped,
        aligned_rat,
        aligned_w,
        aligned_d,
        alpha.view(alpha_shape),
        -2,
    )
    return out.flatten(-2)


def cca_2d_kernel(
    x: Tensor,
    rat: Tensor,
    w: Tensor,
    d: Tensor,
    alpha: Tensor,
    groups: int,
) -> Tensor:
    """Group-native CCA kernel for NCHW feature maps."""
    batch, channels, height, width = x.shape
    grouped = x.reshape(batch, groups, channels // groups, height, width)
    # [B, 5, G] -> [B, 5, G, 1, 1, 1]
    aligned_rat = rat[..., None, None, None]
    aligned_w = w[:, 0, :, None, None, None]
    aligned_d = d[:, 0, :, None, None, None]
    aligned_alpha = alpha.view(1, groups, 1, 1, 1)
    out = _rational_curve(
        grouped, aligned_rat, aligned_w, aligned_d, aligned_alpha, 1
    )
    return out.reshape_as(x)
