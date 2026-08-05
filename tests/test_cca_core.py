import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cca_core import CCA1D, CCA2D, CCALite1D, CCALite2D


def reference_curve(module, x, context):
    """Materializing implementation retained here as an equivalence oracle."""
    rat, w, d = module._gen(context)
    rep = module.feat_dim // module.groups

    if x.ndim in (2, 3):
        expand = lambda value: value.repeat_interleave(rep, dim=-1)
        coefficients = [expand(rat[..., index, :]) for index in range(5)]
        w_expanded, d_expanded = expand(w[..., 0, :]), expand(d[..., 0, :])
        if x.ndim == 3 and coefficients[0].ndim == 2:
            coefficients = [value.unsqueeze(1) for value in coefficients]
            w_expanded, d_expanded = w_expanded.unsqueeze(1), d_expanded.unsqueeze(1)
        alpha = module.alpha.repeat_interleave(rep)
    else:
        batch = x.shape[0]
        expand = lambda value: value[:, 0].repeat_interleave(rep, -1).view(batch, -1, 1, 1)
        coefficients = [expand(rat[:, index:index + 1]) for index in range(5)]
        w_expanded, d_expanded = expand(w), expand(d)
        alpha = module.alpha.repeat_interleave(rep).view(1, -1, 1, 1)
    return module._curve(x, coefficients, w_expanded, d_expanded, alpha)


class CCACoreTests(unittest.TestCase):
    def _randomize(self, module):
        with torch.no_grad():
            module.G[-1].weight.normal_(0, 0.05)
            module.G[-1].bias.normal_(0, 0.05)
            module.theta_0.normal_(0, 0.05)
            module.alpha.normal_(1, 0.05)

    def _check_forward_and_gradients(self, module, x, context):
        self._randomize(module)
        state = {key: value.detach().clone() for key, value in module.state_dict().items()}

        x_fast = x.detach().clone().requires_grad_(True)
        context_fast = context.detach().clone().requires_grad_(True)
        fast = module(x_fast, context_fast)
        fast.square().mean().backward()
        fast_grads = [x_fast.grad.clone(), context_fast.grad.clone()]
        fast_grads += [parameter.grad.clone() for parameter in module.parameters()]

        module.load_state_dict(state)
        module.zero_grad(set_to_none=True)
        x_ref = x.detach().clone().requires_grad_(True)
        context_ref = context.detach().clone().requires_grad_(True)
        ref = reference_curve(module, x_ref, context_ref)
        ref.square().mean().backward()
        ref_grads = [x_ref.grad, context_ref.grad]
        ref_grads += [parameter.grad for parameter in module.parameters()]

        torch.testing.assert_close(fast, ref, rtol=2e-5, atol=2e-6)
        for fast_grad, ref_grad in zip(fast_grads, ref_grads):
            torch.testing.assert_close(fast_grad, ref_grad, rtol=5e-5, atol=2e-6)

    def test_1d_batch_equivalence(self):
        self._check_forward_and_gradients(CCA1D(8, 24, groups=6, g_hidden=7),
                                          torch.randn(5, 24), torch.randn(5, 8))

    def test_1d_sequence_batch_context_equivalence(self):
        self._check_forward_and_gradients(CCA1D(8, 24, groups=6, g_hidden=7),
                                          torch.randn(3, 4, 24), torch.randn(3, 8))

    def test_1d_sequence_token_context_equivalence(self):
        self._check_forward_and_gradients(CCA1D(8, 24, groups=6, g_hidden=7),
                                          torch.randn(3, 4, 24), torch.randn(3, 4, 8))

    def test_2d_equivalence(self):
        module = CCA2D(12, ctx_dim=7, groups=6, g_hidden=5)
        self._check_forward_and_gradients(module, torch.randn(3, 12, 5, 4),
                                          torch.randn(3, 7))

    def test_exact_silu_initialization(self):
        module = CCA1D(8, 32, groups=8)
        x, context = torch.randn(4, 32), torch.randn(4, 8)
        torch.testing.assert_close(module(x, context), torch.nn.functional.silu(x))

    def test_lite_presets(self):
        self.assertEqual(CCALite1D(8, 32).groups, 16)
        self.assertEqual(CCALite2D(24).groups, 12)  # largest divisor <= 16


if __name__ == "__main__":
    unittest.main()
