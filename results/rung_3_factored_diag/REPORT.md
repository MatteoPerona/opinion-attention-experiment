# Task 5 Report — Factored-logit DIAGNOSTIC (not a headline)

> **Label:** architecture ablation outside the run-#1 constraint set.
> Do not cite as the main FJ recovery result.

## Purpose
Isolate whether the rung-3 A_cur↔ΛW gap is caused by the *joint* 2N-way softmax
(if factored recovers near ridge → yes) or by the bilinear ID parameterization itself
(if it doesn't → no).

## Setup
- σ=0.05, M=2000, N=50, d=50, seed=0, epochs=800
- Factored row: mass g_i on own anchor; (1−g_i)·softmax over N current tokens
- Raw-scalar values retained; no W_V / MLP

## Primary comparison

| Method | rel-F(A_cur vs ΛW) | stub corr | val MSE |
|--------|--------------------|-----------|---------|
| Joint 2N-softmax (run #1) | 0.4073 | 0.9995 | — |
| **Factored (this diag)** | **0.0943** | **1.0000** (gates) | 2.5160e-03 |
| Raw ridge | 0.2365 | — | 2.4885e-03 |
| Proj ridge | 0.2190 | — | — |

- Spearman (factored A_cur vs ΛW) = 0.9872
- Gate MAE vs (1−λ) = 0.0014
- Rescaled A_cur→W rel-F = 0.1146
- κ = 1.3883e+01; κ_Z = 1.2618e+03

## Heatmaps
`heatmap_A_vs_W.png`

## Interpretation
Factored variant recovers ΛW near ridge → rung-3 gap was largely caused by the *joint* 2N-way softmax coupling current and anchor tokens.
