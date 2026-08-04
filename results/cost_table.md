# Coste y rentabilidad por tarea (CCA vs SiLU)

| Tarea | Mejora | Params (+%) | FLOPs (+%) | Coste throughput (%) | R = ganancia/coste | ¿Rentable? |
|---|---|---|---|---|---|---|
| Text gen (char) | +47.0% | +17% | +14% | 58% | 0.81 | no |
| Image cls | +20.8% | +124% | +2% | 85% | 0.25 | no |
| Image gen (MNIST) | +15.5% | +71% | +88% | 86% | 0.18 | no |
| Image gen (Fashion) | +23.9% | +71% | +88% | 86% | 0.28 | no |
| Image gen (CIFAR) | +13.7% | +57% | +88% | 86% | 0.16 | no |
| Text cls | +40.0% | +83% | +82% | 84% | 0.48 | no |
| Digits multi-domain | +40.5% | +124% | +2% | 86% | 0.47 | no |
| Algorithmic remap | +37.3% | +78% | +77% | 86% | 0.44 | no |
| Audio digits | +12.3% | +100% | +2% | 82% | 0.15 | no |
| Text sentiment | -4.0% | +31% | +30% | 80% | -0.05 | no |