# Tabla comparativa — CCA vs Baseline (multi-régimen, media ± std)

| Tarea | Reg. | Métrica | Baseline (SiLU) | CCA | CCA-shuffle | Mejora rel. |
|---|---|---|---|---|---|---|
| Text generation | 32 | PPL | 21.920 $\pm$ 0.363 | 11.610 $\pm$ 0.134 | 39.837 | **+47.0%** |
| Image classification | 8 | Acc | 85.047 $\pm$ 0.707 | 88.163 $\pm$ 1.212 | 52.007 $\pm$ 2.813 | **+20.8%** |
| Image generation | 16 | bits/dim | 1.032 $\pm$ 0.006 | 0.872 $\pm$ 0.004 | 3.471 $\pm$ 0.069 | **+15.5%** |
| Text classification | 8 | Acc | 94.271 $\pm$ 2.223 | 96.562 $\pm$ 1.555 | 79.612 $\pm$ 4.268 | **+40.0%** |

*Mejora relativa: reducción de error para accuracy; reducción de la métrica (menor es mejor) para PPL/-ELBO. Accuracy en %.*