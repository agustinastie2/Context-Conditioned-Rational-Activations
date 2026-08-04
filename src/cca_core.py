"""
cca_core.py — Context-Conditioned Rational Activation (CCA), shared module.
==========================================================================

Single source of truth for the CCA layer used across all modalities
(text gen/cls, image cls/gen). Two variants share the same maths and the same
instrumentation:

    CCA1D : x is [B, F] or [B, S, F]   (vectors / token sequences)
    CCA2D : x is [B, C, H, W]          (conv feature maps)

Maths (per channel-group g):

    A(x, c) = x * sigmoid(x)
            + alpha_g * (a0 + a1*x + a2*x^2) / (1 + (b0 + b1*x)^2)

    theta(c) = [a0, a1, a2, b0, b1] = theta_0 + gamma * tanh(G(c))

Design choices that make it work in practice (found by the sweeps in this
repo; see PAPER.md §3 for the rationale):

  * theta_0 = 0 and G's LAST layer zero-initialised  ->  the layer is *exactly*
    SiLU at init, so adding CCA never hurts a converged baseline at step 0.
  * G is a 2-layer MLP (depth helps the generator separate regimes).
  * Coefficients are produced per CHANNEL-GROUP (not one global scalar set),
    giving the curve enough capacity to specialise different channels.
  * alpha is learnable per group (the network tunes how much rational
    correction to mix in); gamma is a fixed modulation gain.

The context `c` is always supplied by the host model — typically a global
pooling of the previous hidden layer, or an external (strictly-past) summary.
The layer never looks at the future; causality is the caller's contract.
"""

import torch
import torch.nn as nn

import os as _os

DEFAULT_GROUPS = int(_os.environ.get("CCA_GROUPS", 64))
DEFAULT_ALPHA = 1.0
DEFAULT_GAMMA = 2.0
DEFAULT_GAMMA_AFF = 1.0     # modulation gain for the context-dependent affine
G_HIDDEN = int(_os.environ.get("CCA_GHID", 64))

# Cost knobs (the "CCA-lite" Pareto study in cca_efficiency.py drives these).
# All default to the tuned full config, so behaviour is unchanged unless set:
#   CCA_GROUPS   coefficient granularity  (64 -> 16/8 makes the generator cheaper)
#   CCA_GHID     generator hidden width   (64 -> 16)
#   CCA_LOWRANK  low-rank factor r for G's last layer (0 = off / full rank)
# A smaller generator keeps the exact-SiLU start (its last layer is still
# zero-initialised) and the same maths; only its capacity/throughput change.
DEFAULT_LOWRANK = int(_os.environ.get("CCA_LOWRANK", 0))

# v2 adds a context-dependent input affine (scale w, shift d) so the curve can
# RE-CENTRE per regime, not just scale a correction. With theta_0 = 0 and G's
# last layer zero-init, at step 0 w=1, d=0 and the rational term is 0, so the
# layer is still *exactly* SiLU. Set CCA_AFFINE=0 to recover the v1 behaviour.
USE_AFFINE = _os.environ.get("CCA_AFFINE", "1") != "0"


class _CCABase(nn.Module):
    def __init__(self, ctx_dim, feat_dim, groups, alpha, gamma,
                 affine=USE_AFFINE, gamma_aff=DEFAULT_GAMMA_AFF,
                 g_hidden=None, lowrank=None):
        super().__init__()
        assert feat_dim % groups == 0, (feat_dim, groups)
        self.groups, self.feat_dim, self.gamma = groups, feat_dim, gamma
        self.affine, self.gamma_aff = affine, gamma_aff
        self.n_out = 7 if affine else 5         # +2 for (w, d)
        gh = G_HIDDEN if g_hidden is None else g_hidden
        lr = DEFAULT_LOWRANK if lowrank is None else lowrank
        out = self.n_out * groups
        # 2-layer generator; last layer zero-init -> exact SiLU at start.
        # Optional low-rank factorisation of the (wide) output projection keeps
        # the same start and maths but cuts its parameter/throughput cost.
        gen = [nn.Linear(ctx_dim, gh), nn.SiLU()]
        if lr and lr < out:
            gen += [nn.Linear(gh, lr, bias=False), nn.Linear(lr, out)]
        else:
            gen += [nn.Linear(gh, out)]
        self.G = nn.Sequential(*gen)
        nn.init.zeros_(self.G[-1].weight)
        nn.init.zeros_(self.G[-1].bias)
        self.theta_0 = nn.Parameter(torch.zeros(5 * groups))
        self.alpha = nn.Parameter(torch.full((groups,), float(alpha)))
        # instrumentation
        self.shuffle_ctx = False     # ablation: permute context across batch
        self.collect = False         # accumulate coefficients for variance
        self._buf = []

    # -- instrumentation ---------------------------------------------------- #
    def reset_stats(self):
        self._buf = []

    def coeff_variance(self):
        """Per-coefficient variance across samples, averaged over groups.
        Returns the 5 rational coeffs (a0,a1,a2,b0,b1) for back-compatibility."""
        if not self._buf:
            return None
        c = torch.cat(self._buf, 0)                            # [N, n_out, G]
        return c[:, :5].var(dim=0, unbiased=False).mean(dim=1).cpu()   # [5]

    def coeff_mean(self):
        if not self._buf:
            return None
        c = torch.cat(self._buf, 0)
        return c.mean(dim=0).cpu()                             # [n_out, G]

    # -- shared coefficient generation -------------------------------------- #
    def _gen(self, context):
        """context [*L, ctx_dim] -> (rational [*L,5,G], w [*L,1,G], d [*L,1,G]).
        Leading dims *L are preserved, so the context may be one vector per
        example ([B, ctx]) or one per token ([B, S, ctx])."""
        if self.shuffle_ctx:
            context = context[torch.randperm(context.size(0), device=context.device)]
        lead = context.shape[:-1]
        raw = self.G(context).view(*lead, self.n_out, self.groups)
        if self.collect:
            self._buf.append(raw.reshape(-1, self.n_out, self.groups).detach())
        theta0 = self.theta_0.view(*([1] * len(lead)), 5, self.groups)
        rat = theta0 + self.gamma * torch.tanh(raw[..., :5, :])
        if self.affine:
            w = 1.0 + self.gamma_aff * torch.tanh(raw[..., 5:6, :])   # init 1
            d = self.gamma_aff * torch.tanh(raw[..., 6:7, :])         # init 0
        else:
            w = torch.ones_like(rat[..., :1, :])
            d = torch.zeros_like(rat[..., :1, :])
        return rat, w, d

    def _curve(self, x, rat, w, d, alpha_b):
        """Apply the activation given coeffs already broadcast to x's shape."""
        a0, a1, a2, b0, b1 = rat
        u = w * x + d
        silu = u * torch.sigmoid(u)
        return silu + alpha_b * (a0 + a1 * u + a2 * u * u) / (
            1 + (b0 + b1 * u) ** 2)


class CCA1D(_CCABase):
    """x: [B, F] or [B, S, F].  context: [B, ctx_dim]."""

    def __init__(self, ctx_dim, feat_dim, groups=DEFAULT_GROUPS,
                 alpha=DEFAULT_ALPHA, gamma=DEFAULT_GAMMA, **kw):
        super().__init__(ctx_dim, feat_dim, groups, alpha, gamma, **kw)

    def forward(self, x, context):
        rat, w, d = self._gen(context)             # leading dims = ctx.shape[:-1]
        rep = self.feat_dim // self.groups

        def exp(t):                                # [*L, G] -> [*L, F]
            return t.repeat_interleave(rep, dim=-1)
        rat5 = [exp(rat[..., i, :]) for i in range(5)]
        w_, d_ = exp(w[..., 0, :]), exp(d[..., 0, :])
        # if context is per-example ([B,ctx]) but x is a sequence ([B,S,F]),
        # insert a singleton sequence axis so the coefficients broadcast over S.
        if x.dim() == 3 and rat5[0].dim() == 2:
            rat5 = [t.unsqueeze(1) for t in rat5]
            w_, d_ = w_.unsqueeze(1), d_.unsqueeze(1)
        alpha_b = self.alpha.repeat_interleave(rep)        # [F], broadcasts
        return self._curve(x, rat5, w_, d_, alpha_b)


class CCA2D(_CCABase):
    """x: [B, C, H, W].  context defaults to global-average-pool of x."""

    def __init__(self, channels, ctx_dim=None, groups=DEFAULT_GROUPS,
                 alpha=DEFAULT_ALPHA, gamma=DEFAULT_GAMMA, **kw):
        groups = min(groups, channels)
        while channels % groups:
            groups -= 1
        super().__init__(ctx_dim or channels, channels, groups, alpha, gamma, **kw)

    def forward(self, x, context=None):
        if context is None:
            context = x.mean(dim=(2, 3))                       # GAP -> [B, C]
        rat, w, d = self._gen(context)
        rep = self.feat_dim // self.groups
        B = x.size(0)

        def exp(t):                                # [B,1,G] -> [B,C,1,1]
            return t[:, 0].repeat_interleave(rep, -1).view(B, -1, 1, 1)
        rat5 = [exp(rat[:, i:i + 1]) for i in range(5)]
        w_, d_ = exp(w), exp(d)
        alpha_b = self.alpha.repeat_interleave(rep).view(1, -1, 1, 1)
        return self._curve(x, rat5, w_, d_, alpha_b)


# --------------------------------------------------------------------------- #
#  helpers shared by host models
# --------------------------------------------------------------------------- #
def cca_modules(model):
    return [m for m in model.modules() if isinstance(m, _CCABase)]


def set_cca(model, *, collect=False, shuffle_ctx=False):
    for m in cca_modules(model):
        m.collect = collect
        m.shuffle_ctx = shuffle_ctx
        if collect:
            m.reset_stats()


def mean_coeff_variance(model):
    vs = [m.coeff_variance() for m in cca_modules(model)]
    vs = [v for v in vs if v is not None]
    if not vs:
        return None
    return torch.stack(vs).mean(0)                            # [5]


def curve_at(model, x_grid, contexts):
    """
    Evaluate the learned 1-D activation curve of the FIRST CCA module for a set
    of context vectors. Returns y of shape [n_ctx, len(x_grid)] (group 0).
    Used to plot how the activation morphs with context.
    """
    mod = cca_modules(model)[0]
    x = x_grid.view(1, -1).to(next(model.parameters()).device)
    ys = []
    was = (mod.collect, mod.shuffle_ctx)
    mod.collect = mod.shuffle_ctx = False
    with torch.no_grad():
        for c in contexts:
            rat, w, d = mod._gen(c.view(1, -1).to(x.device))
            a0, a1, a2, b0, b1 = rat[0, :, 0].tolist()   # group 0
            wv, dv = w[0, 0, 0].item(), d[0, 0, 0].item()
            u = wv * x + dv
            silu = u * torch.sigmoid(u)
            y = silu + mod.alpha[0].item() * (a0 + a1 * u + a2 * u * u) / (
                1 + (b0 + b1 * u) ** 2)
            ys.append(y.squeeze(0).cpu())
    mod.collect, mod.shuffle_ctx = was
    return torch.stack(ys)
