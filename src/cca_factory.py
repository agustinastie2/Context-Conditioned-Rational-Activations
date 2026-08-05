"""Environment-selectable CCA construction for the original task suite.

Defaults preserve the original full CCA exactly. The same host architectures
can also instantiate every paper baseline through ``CCA_VARIANT`` so all
methods receive precisely the context used by CCA.
"""

import os

from adaptive_activations import ActivationConfig, build_activation
from cca_core import CCA1D, CCA2D, CCALite1D, CCALite2D


def make_cca1d(ctx_dim, feat_dim, **kwargs):
    variant = os.environ.get("CCA_VARIANT", "full").lower()
    groups = int(os.environ.get("ACT_GROUPS", "16"))
    hidden = int(os.environ.get("ACT_HIDDEN", "16"))
    if variant in ("hinge", "cca_hinge"):
        return build_activation(ActivationConfig("cca_hinge", groups, hidden),
                                ctx_dim, feat_dim)
    if variant in ("lite", "cca_lite"):
        return CCALite1D(ctx_dim, feat_dim, **kwargs)
    if variant in ("rational", "dyrelu", "meta_acon"):
        return build_activation(ActivationConfig(variant, groups, hidden),
                                ctx_dim, feat_dim)
    if variant == "full":
        return CCA1D(ctx_dim, feat_dim, **kwargs)
    raise ValueError(f"Unknown CCA_VARIANT={variant!r}; use full, lite, hinge, "
                     "rational, dyrelu, or meta_acon")


def make_cca2d(channels, ctx_dim=None, **kwargs):
    variant = os.environ.get("CCA_VARIANT", "full").lower()
    groups = int(os.environ.get("ACT_GROUPS", "16"))
    hidden = int(os.environ.get("ACT_HIDDEN", "16"))
    if variant in ("hinge", "cca_hinge"):
        return build_activation(ActivationConfig("cca_hinge", groups, hidden),
                                ctx_dim or channels, channels, spatial=True)
    if variant in ("lite", "cca_lite"):
        return CCALite2D(channels, ctx_dim=ctx_dim, **kwargs)
    if variant in ("rational", "dyrelu", "meta_acon"):
        return build_activation(ActivationConfig(variant, groups, hidden),
                                ctx_dim or channels, channels, spatial=True)
    if variant == "full":
        return CCA2D(channels, ctx_dim=ctx_dim, **kwargs)
    raise ValueError(f"Unknown CCA_VARIANT={variant!r}; use full, lite, hinge, "
                     "rational, dyrelu, or meta_acon")
