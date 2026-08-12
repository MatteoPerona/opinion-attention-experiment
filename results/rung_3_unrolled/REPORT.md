# Task 4 Report — Unrolled FJ training (§4 graduation path)

## Setup
- Rung-3 config: dense FJ + 2N anchors, σ=0.05, M=2000, N=50, T=12, d=50
- Curriculum: k=1 (300 ep) → k=2 (200) → k=4 (200); grad clip=1.0
- Anchors fixed at true x(0) throughout rollout; raw-scalar values; no new modules
- Seeds=[0, 1, 2]

## Success criterion (fixed before running)
Success if mean unrolled attn rel-F(A_cur vs ΛW) improves on one-step ≈0.40 by a margin larger than seed spread ≈0.05 (i.e. mean < 0.35); interesting if approaching ridge ≈0.23. Stubbornness corr must stay >0.99.

## One-step vs unrolled (primary: rel-F A_cur vs ΛW)

| | Attention rel-F | Ridge rel-F | Stub corr |
|--|-----------------|-------------|-----------|
| Run #1 one-step (seed 0) | 0.4073 | 0.2365 | 0.9995 |
| Run #1 one-step (3 seeds) | [0.407, 0.37, 0.421] | — | — |
| **Unrolled mean** | **0.4094** | — | — |

| seed | unrolled attn | raw ridge | proj ridge | stub corr | attn MSE | κ | κ_Z | unstable |
|------|---------------|-----------|------------|-----------|----------|---|-----|----------|
| 0 | 0.4237 | 0.2365 | 0.2190 | 0.9998 | 2.5362e-03 | 1.388e+01 | 1.262e+03 | False |
| 1 | 0.3911 | 0.2224 | 0.2062 | 0.9998 | 2.5458e-03 | 1.410e+01 | 1.128e+03 | False |
| 2 | 0.4135 | 0.2144 | 0.1984 | 0.9998 | 2.5398e-03 | 1.163e+01 | 1.157e+03 | False |

## Prediction MSE / condition numbers
See table; artifacts in `metrics.json`.

## Heatmaps / loss curves
`heatmap_A_vs_W.png`; per-seed `loss_curves.png`.

## Verdict
NO SUCCESS on criterion: mean unrolled rel-F=0.4094 did not beat one-step 0.40 by >0.05. Stubbornness preserved (corr>0.99).
