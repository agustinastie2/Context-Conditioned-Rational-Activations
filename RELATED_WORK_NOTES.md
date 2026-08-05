# Related-work notes for the 10-page extension

This is a source dossier for a later paper revision. It is intentionally kept
separate from `PAPER.md` and the LaTeX manuscripts.

## Direct comparators

### Dynamic ReLU

Chen et al., *Dynamic ReLU*, ECCV 2020, introduce a dynamic piecewise-linear
rectifier whose slopes and intercepts are generated from global input context.
The closest comparator to CCA is DY-ReLU-B: channel-specific parameters and two
linear branches. The confirmatory implementation uses two grouped branches,
starts as exact ReLU, and supplies the same context used by CCA rather than
domain metadata.

- Primary source: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3223_ECCV_2020_paper.php
- Mechanistic distinction: both methods use a context hypernetwork; Dynamic
  ReLU selects a piecewise-linear envelope, while CCA generates a smooth affine
  plus rational correction.

### Meta-ACON

Ma et al., *Activate or Not: Learning Customized Activation*, CVPR 2021,
formulate ACON as a smooth interpolation between two learned linear responses.
Meta-ACON makes the switching factor input-dependent. The confirmatory version
uses grouped learned `p1/p2` values and a context-generated beta initialized to
the Swish/SiLU regime.

- Primary source: https://openaccess.thecvf.com/content/CVPR2021/html/Ma_Activate_or_Not_Learning_Customized_Activation_CVPR_2021_paper.html
- Mechanistic distinction: Meta-ACON conditions the nonlinear switching
  strength; CCA conditions seven curve/affine coefficients per group.

### Rational activations

Boullé, Nakatsukasa, and Townsend, *Rational Neural Networks*, NeurIPS 2020,
establish approximation advantages for trainable rational activations. Molina,
Schramowski, and Kersting, *Padé Activation Units*, ICLR 2020, demonstrate
end-to-end learned rational functions in deep networks. The confirmatory static
baseline uses exactly CCA's stabilized low-degree rational family but removes
the context generator. This isolates conditioning from rationality without
introducing a different denominator or numerical-stability mechanism.

- NeurIPS source: https://proceedings.neurips.cc/paper/2020/hash/a3f390d88e4c41f2747bfa2f1b5f87db-Abstract.html
- ICLR source: https://openreview.net/forum?id=BJlBSkHtDS
- Mechanistic distinction: the static rational curve is learned once per group;
  CCA produces sample-dependent coefficients at inference.

## Fair-comparison protocol

- CCA-Lite with the original backbone defines each task's resource target.
- Competing architectures are chosen only by parameter and analytical-FLOP
  proximity, before training and without inspecting validation/test accuracy.
- Every budget-controlled model is below 105% of target parameters and 110% of
  target FLOPs; exact utilization is recorded in `budgets.json`.
- Training optimizer, learning rate, batch size, steps/epochs, splits, and paired
  seeds are identical across methods.
- Adaptive competitors receive the same inferred feature context. Domain IDs
  are used only after prediction to calculate per-domain metrics.

## Evidence to integrate later

Use `results/extended_comparison/EVIDENCE.md` for the numeric tables and
`REPRODUCIBILITY.md` for dataset, split, architecture, hyperparameter, context,
statistical-test, and execution details. The confirmatory result supports a
careful claim: CCA-Lite clearly improves over fixed SiLU and static rational
activations in the real multi-domain task, but is statistically tied with the
strong Dynamic ReLU-B comparator under the matched budget.

## Expressivity follow-up

A validation-only search on seeds 2001–2003 selected **CCA-Hinge**, which adds
the smooth context-conditioned basis
`lambda(c) * sqrt((u - t(c))^2 + epsilon)`. A two-branch maximum can be written
as a linear component plus an absolute-value hinge, so this term supplies the
piecewise-linear capability missing from the low-degree rational curve. Its
zero-initialized amplitude preserves exact-SiLU initialization.

On untouched seeds 3001–3010, CCA-Hinge achieves 88.68% ± 0.50 on real
MNIST+SVHN+USPS versus 88.06% ± 0.98 for budget-matched Dynamic ReLU-B: a paired
+0.62 percentage-point gain (95% CI +0.21 to +1.11, Holm-corrected p=0.02344).
It remains tied with Dynamic ReLU-B on synthetic text. Full results are in
`results/cca_hinge_confirmation/EVIDENCE.md`.
