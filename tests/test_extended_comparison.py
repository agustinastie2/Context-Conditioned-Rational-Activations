import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from adaptive_activations import (
    ActivationConfig, ContextSiLU, DynamicReLU, MetaACON, StaticRational,
    build_activation,
)
from extended_comparison import (
    DigitModel, ModelConfig, TextModel, exact_paired_permutation,
    holm_adjust, resolve_budgets,
)
from cca_factory import make_cca1d, make_cca2d
from cca_core import CCA1D, CCA2D, cca_modules


class ActivationComparisonTests(unittest.TestCase):
    def test_initialization_contracts(self):
        x = torch.randn(4, 32)
        context = torch.randn(4, 8)
        torch.testing.assert_close(ContextSiLU()(x, context), torch.nn.functional.silu(x))
        torch.testing.assert_close(StaticRational(32)(x, context), torch.nn.functional.silu(x))
        torch.testing.assert_close(DynamicReLU(8, 32)(x, context), torch.relu(x))
        meta = MetaACON(8, 32)
        torch.testing.assert_close(meta(x, context), torch.nn.functional.silu(x))
        hinge = build_activation(ActivationConfig("cca_hinge", groups=8, g_hidden=32), 8, 32)
        torch.testing.assert_close(hinge(x, context), torch.nn.functional.silu(x))

    def test_all_methods_shapes_and_gradients(self):
        for spatial, shape, ctx_dim in ((False, (3, 32), 8), (True, (3, 16, 7, 7), 16)):
            x = torch.randn(shape, requires_grad=True)
            context = torch.randn(shape[0], ctx_dim, requires_grad=True)
            for method in ("silu", "silu_budget", "rational", "dyrelu", "meta_acon", "cca_lite"):
                activation = build_activation(
                    ActivationConfig(method), ctx_dim, shape[1] if spatial else shape[-1],
                    spatial=spatial,
                )
                output = activation(x, context)
                self.assertEqual(output.shape, x.shape)
                output.mean().backward(retain_graph=True)
                self.assertTrue(torch.isfinite(output).all())

    def test_static_rational_ignores_context(self):
        activation = StaticRational(16)
        x = torch.randn(2, 16)
        torch.testing.assert_close(activation(x, torch.randn(2, 4)),
                                   activation(x, torch.randn(2, 4)))

    def test_models_expose_expected_outputs(self):
        text_model = TextModel(ModelConfig("cca_lite", hidden=256))
        digit_model = DigitModel(ModelConfig("cca_lite", channels1=16, channels2=32))
        self.assertEqual(text_model(torch.randint(0, 27, (2, 48))).shape, (2, 4))
        self.assertEqual(digit_model(torch.randn(2, 1, 28, 28)).shape, (2, 10))

    def test_budget_resolver_enforces_caps(self):
        budgets = resolve_budgets()
        for task in budgets.values():
            for method, entry in task["methods"].items():
                if method != "silu":
                    self.assertTrue(entry["within_budget"])
                    self.assertLessEqual(entry["params_vs_target_pct"], 5.0 + 1e-9)
                    self.assertLessEqual(entry["flops_vs_target_pct"], 10.0 + 1e-9)

    def test_exact_statistics(self):
        self.assertEqual(exact_paired_permutation([1.0, 1.0]), 0.5)
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.2})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.08)
        self.assertAlmostEqual(adjusted["c"], 0.2)

    def test_factory_preserves_default_and_exposes_hinge(self):
        import os
        previous = os.environ.get("CCA_VARIANT")
        try:
            os.environ.pop("CCA_VARIANT", None)
            self.assertIsInstance(make_cca1d(8, 64), CCA1D)
            self.assertIsInstance(make_cca2d(16), CCA2D)
            os.environ["CCA_VARIANT"] = "hinge"
            one_d, two_d = make_cca1d(8, 64), make_cca2d(16)
            self.assertEqual(len(cca_modules(torch.nn.Sequential(one_d, two_d))), 2)
            x = torch.randn(2, 64)
            torch.testing.assert_close(one_d(x, torch.randn(2, 8)),
                                       torch.nn.functional.silu(x))
        finally:
            if previous is None:
                os.environ.pop("CCA_VARIANT", None)
            else:
                os.environ["CCA_VARIANT"] = previous

    def test_factory_exposes_all_paper_baselines_with_shared_spatial_context(self):
        import os
        previous = os.environ.get("CCA_VARIANT")
        try:
            x1 = torch.randn(2, 32, requires_grad=True)
            context = torch.randn(2, 8, requires_grad=True)
            x2 = torch.randn(2, 16, 5, 5, requires_grad=True)
            for method in ("rational", "dyrelu", "meta_acon", "cca_lite", "cca_hinge"):
                os.environ["CCA_VARIANT"] = method
                one_d, two_d = make_cca1d(8, 32), make_cca2d(16)
                y1, y2 = one_d(x1, context), two_d(x2)
                self.assertEqual(y1.shape, x1.shape)
                self.assertEqual(y2.shape, x2.shape)
                (y1.mean() + y2.mean()).backward(retain_graph=True)
                self.assertTrue(torch.isfinite(y1).all() and torch.isfinite(y2).all())
        finally:
            if previous is None:
                os.environ.pop("CCA_VARIANT", None)
            else:
                os.environ["CCA_VARIANT"] = previous


if __name__ == "__main__":
    unittest.main()
