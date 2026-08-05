"""Comparable activation families for the confirmatory CCA study.

Every activation exposes ``forward(x, context=None)``. Adaptive methods receive
the exact same context tensor from the host model; static methods ignore it.
All channel-wise parameters remain grouped and broadcast without materializing
per-feature copies, making the implementations friendly to ``torch.compile``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from cca_core import CCALite1D, CCALite2D


METHODS = ("silu", "silu_budget", "rational", "dyrelu", "meta_acon", "cca_lite")


def valid_groups(features: int, requested: int) -> int:
    groups = min(features, requested)
    while features % groups:
        groups -= 1
    return groups


def _group_view(x: Tensor, groups: int, spatial: bool) -> Tensor:
    if spatial:
        batch, channels, height, width = x.shape
        return x.reshape(batch, groups, channels // groups, height, width)
    return x.unflatten(-1, (groups, x.shape[-1] // groups))


def _expand_group(value: Tensor, x: Tensor, spatial: bool) -> Tensor:
    """Broadcast ``[*context,G]`` against a grouped input tensor."""
    if spatial:
        return value[:, :, None, None, None]
    # Insert axes absent from per-example context (e.g. the token axis).
    target_ndim = x.ndim
    for _ in range(target_ndim - value.ndim):
        value = value.unsqueeze(-2)
    return value.unsqueeze(-1)


class ContextSiLU(nn.Module):
    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        return torch.nn.functional.silu(x)


class _ContextGenerator(nn.Module):
    def __init__(self, ctx_dim: int, hidden: int, outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.SiLU(), nn.Linear(hidden, outputs)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: Tensor) -> Tensor:
        return self.net(context)


class DynamicReLU(nn.Module):
    """Two-branch Dynamic ReLU-B with grouped, context-generated parameters.

    At initialization the branches are ``x`` and ``0``, hence the module starts
    as exact ReLU. Context changes both slopes and intercepts per sample/group.
    """

    def __init__(self, ctx_dim: int, features: int, *, groups: int = 16,
                 g_hidden: int = 16, spatial: bool = False):
        super().__init__()
        self.features, self.groups = features, valid_groups(features, groups)
        self.spatial = spatial
        self.generator = _ContextGenerator(ctx_dim, g_hidden, 4 * self.groups)
        base = torch.zeros(4, self.groups)
        base[0] = 1.0
        self.register_buffer("base", base)
        self.register_buffer(
            "scale", torch.tensor([1.0, 1.0, 0.5, 0.5]).view(4, 1)
        )

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        if context is None and self.spatial:
            context = x.mean(dim=(2, 3))
        if context is None:
            raise ValueError("DynamicReLU requires context")
        lead = context.shape[:-1]
        dynamic = self.generator(context).view(*lead, 4, self.groups)
        params = self.base.view(*([1] * len(lead)), 4, self.groups)
        params = params + self.scale.view(*([1] * len(lead)), 4, 1) * torch.tanh(dynamic)
        grouped = _group_view(x, self.groups, self.spatial)
        a1, a2, b1, b2 = params.unbind(-2)
        a1, a2 = _expand_group(a1, x, self.spatial), _expand_group(a2, x, self.spatial)
        b1, b2 = _expand_group(b1, x, self.spatial), _expand_group(b2, x, self.spatial)
        out = torch.maximum(a1 * grouped + b1, a2 * grouped + b2)
        return out.reshape_as(x)


class MetaACON(nn.Module):
    """Grouped Meta-ACON with context-generated switching factor beta."""

    def __init__(self, ctx_dim: int, features: int, *, groups: int = 16,
                 g_hidden: int = 16, spatial: bool = False):
        super().__init__()
        self.features, self.groups = features, valid_groups(features, groups)
        self.spatial = spatial
        self.p1 = nn.Parameter(torch.ones(self.groups))
        self.p2 = nn.Parameter(torch.zeros(self.groups))
        self.generator = _ContextGenerator(ctx_dim, g_hidden, self.groups)

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        if context is None and self.spatial:
            context = x.mean(dim=(2, 3))
        if context is None:
            raise ValueError("MetaACON requires context")
        grouped = _group_view(x, self.groups, self.spatial)
        p1 = _expand_group(self.p1.expand(*context.shape[:-1], -1), x, self.spatial)
        p2 = _expand_group(self.p2.expand(*context.shape[:-1], -1), x, self.spatial)
        beta_raw = self.generator(context)
        beta = _expand_group(2.0 * torch.sigmoid(beta_raw), x, self.spatial)
        delta = (p1 - p2) * grouped
        out = delta * torch.sigmoid(beta * delta) + p2 * grouped
        return out.reshape_as(x)


class StaticRational(nn.Module):
    """Context-independent counterpart of CCA's stabilized rational family."""

    def __init__(self, features: int, *, groups: int = 16, spatial: bool = False):
        super().__init__()
        self.features, self.groups = features, valid_groups(features, groups)
        self.spatial = spatial
        self.theta = nn.Parameter(torch.zeros(5, self.groups))
        self.w = nn.Parameter(torch.ones(self.groups))
        self.d = nn.Parameter(torch.zeros(self.groups))
        self.alpha = nn.Parameter(torch.ones(self.groups))

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        grouped = _group_view(x, self.groups, self.spatial)
        lead = (x.shape[0],) if self.spatial else x.shape[:-1]
        def expand(value: Tensor) -> Tensor:
            return _expand_group(value.expand(*lead, -1), x, self.spatial)
        a0, a1, a2, b0, b1 = (expand(value) for value in self.theta.unbind(0))
        w, d, alpha = expand(self.w), expand(self.d), expand(self.alpha)
        u = w * grouped + d
        numerator = a0 + a1 * u + a2 * u * u
        denominator_base = b0 + b1 * u
        out = torch.nn.functional.silu(u) + alpha * numerator / (
            1.0 + denominator_base * denominator_base
        )
        return out.reshape_as(x)


class CCAHinge(nn.Module):
    """CCA plus a smooth context-conditioned hinge (Maxout-like) residual.

    ``lambda(c) * sqrt((u - t(c))^2 + eps)`` supplies the kink-like basis that
    a low-degree rational curve lacks. Since ``lambda(c)=0`` at initialization,
    the complete activation still starts as exact SiLU while gradients reach
    the hinge amplitude immediately.
    """

    def __init__(self, ctx_dim: int, features: int, *, groups: int = 8,
                 g_hidden: int = 32, spatial: bool = False, eps: float = 1e-4):
        super().__init__()
        self.features, self.groups = features, valid_groups(features, groups)
        self.spatial, self.eps = spatial, eps
        self.generator = _ContextGenerator(ctx_dim, g_hidden, 9 * self.groups)
        self.theta_0 = nn.Parameter(torch.zeros(5, self.groups))
        self.alpha = nn.Parameter(torch.ones(self.groups))
        self.shuffle_ctx = False
        self.collect = False
        self._buf = []
        self._is_cca_module = True

    def reset_stats(self):
        self._buf = []

    def coeff_variance(self):
        if not self._buf:
            return None
        coefficients = torch.cat(self._buf, 0)
        return coefficients.var(dim=0, unbiased=False).mean(dim=1).cpu()

    def curve_for_context(self, x_grid: Tensor, context: Tensor, group: int = 0) -> Tensor:
        """Evaluate one learned group for plotting/diagnostics."""
        raw = self.generator(context).view(*context.shape[:-1], 9, self.groups)
        bounded = torch.tanh(raw)
        rat = self.theta_0 + 2.0 * bounded[..., :5, :]
        a0, a1, a2, b0, b1 = rat[0, :, group]
        w = 1.0 + bounded[0, 5, group]
        d = bounded[0, 6, group]
        amplitude = bounded[0, 7, group]
        threshold = bounded[0, 8, group]
        u = w * x_grid + d
        rational = self.alpha[group] * (a0 + a1 * u + a2 * u.square()) / (
            1.0 + (b0 + b1 * u).square()
        )
        return torch.nn.functional.silu(u) + rational + amplitude * torch.sqrt(
            (u - threshold).square() + self.eps
        )

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        if context is None and self.spatial:
            context = x.mean(dim=(2, 3))
        if context is None:
            raise ValueError("CCAHinge requires context")
        if self.shuffle_ctx:
            context = context[torch.randperm(context.shape[0], device=context.device)]
        lead = context.shape[:-1]
        raw = self.generator(context).view(*lead, 9, self.groups)
        bounded = torch.tanh(raw)
        theta0 = self.theta_0.view(*([1] * len(lead)), 5, self.groups)
        rat = theta0 + 2.0 * bounded[..., :5, :]
        if self.collect:
            self._buf.append(rat.detach().reshape(-1, 5, self.groups))
        w = 1.0 + bounded[..., 5, :]
        d = bounded[..., 6, :]
        amplitude = bounded[..., 7, :]
        threshold = bounded[..., 8, :]
        grouped = _group_view(x, self.groups, self.spatial)
        a0, a1, a2, b0, b1 = rat.unbind(-2)
        a0, a1, a2, b0, b1 = (
            _expand_group(value, x, self.spatial) for value in (a0, a1, a2, b0, b1)
        )
        w, d, amplitude, threshold = (
            _expand_group(value, x, self.spatial)
            for value in (w, d, amplitude, threshold)
        )
        alpha = _expand_group(self.alpha.expand(*lead, -1), x, self.spatial)
        u = w * grouped + d
        numerator = a0 + a1 * u + a2 * u * u
        denominator_base = b0 + b1 * u
        rational = alpha * numerator / (1.0 + denominator_base * denominator_base)
        hinge = amplitude * torch.sqrt((u - threshold).square() + self.eps)
        return (torch.nn.functional.silu(u) + rational + hinge).reshape_as(x)


@dataclass(frozen=True)
class ActivationConfig:
    method: str
    groups: int = 16
    g_hidden: int = 16
    lowrank: int = 0


def build_activation(config: ActivationConfig, ctx_dim: int, features: int,
                     *, spatial: bool = False) -> nn.Module:
    method = config.method
    if method in ("silu", "silu_budget"):
        return ContextSiLU()
    if method == "rational":
        return StaticRational(features, groups=config.groups, spatial=spatial)
    if method == "dyrelu":
        return DynamicReLU(ctx_dim, features, groups=config.groups,
                           g_hidden=config.g_hidden, spatial=spatial)
    if method == "meta_acon":
        return MetaACON(ctx_dim, features, groups=config.groups,
                        g_hidden=config.g_hidden, spatial=spatial)
    if method == "cca_lite":
        cls = CCALite2D if spatial else CCALite1D
        if spatial:
            return cls(features, ctx_dim=ctx_dim, groups=config.groups,
                       g_hidden=config.g_hidden, lowrank=config.lowrank)
        return cls(ctx_dim, features, groups=config.groups, g_hidden=config.g_hidden,
                   lowrank=config.lowrank)
    if method == "cca_hinge":
        return CCAHinge(ctx_dim, features, groups=config.groups,
                        g_hidden=config.g_hidden, spatial=spatial)
    raise ValueError(f"Unknown activation method: {method}")
