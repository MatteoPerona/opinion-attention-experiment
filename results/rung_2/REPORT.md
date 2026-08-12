# Rung 2 Report (Run #2) — DeGroot + process noise, 3 seeds + projected ridge

## Setup
- Identical to run #1 rung 2: σ=0.05, N=50, M=200, T=12, d=50, dense DeGroot
- Seeds=[0, 1, 2], epochs=800, lr=0.005
- Comparators: attention, **raw ridge**, **Euclidean simplex-projected ridge**

## Run #1 continuity (archived seed-0 numbers before overwrite)
- Attention rel-F = 0.189623
- Raw ridge rel-F = 0.200684
- (No projected-ridge comparator in run #1)

## Decision rule (fixed before results)
Confirm run-#1 claim "attention matches or beats ridge under noise" **only if**
mean attention rel-F ≤ mean *projected*-ridge rel-F across 3 seeds.
Otherwise reclassify: advantage was the row-stochasticity constraint itself.

## Primary metric (§5 — dense: relative Frobenius)

| seed | attention | raw ridge | simplex-proj ridge | attn val MSE | κ |
|------|-----------|-----------|--------------------|--------------|---|
| 0 | 0.189623 | 0.200684 | 0.194526 | 2.580179e-03 | 6.1350e+01 |
| 1 | 0.188840 | 0.195775 | 0.189800 | 2.598360e-03 | 4.4788e+01 |
| 2 | 0.184489 | 0.191785 | 0.186017 | 2.578076e-03 | 4.8923e+01 |
| **mean** | **0.187650** | **0.196081** | **0.190114** | — | — |

## Prediction MSE (held-out, seed 0)
- Attention = **2.580179e-03**
- Raw ridge = 2.590654e-03
- Simplex-proj ridge = 2.585512e-03

## State-covariance condition number (seed 0)
- κ(cov(X)) = **6.135020e+01**

## Heatmaps
`heatmap_A_vs_W.png`; per-seed also has `heatmap_ridge_simplex_vs_W.png`.

## Verdict
**CONFIRMED:** mean attention rel-F (0.187650) ≤ mean simplex-projected ridge rel-F (0.190114). Attention matches or beats the fair (row-stochastic) ridge comparator under noise.
