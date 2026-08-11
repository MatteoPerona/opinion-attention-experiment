# Rung 2 Report — DeGroot + process noise

## Setup
- Exactly one change vs rung 1: **σ = 0.05** process noise
- N=50, M=200, T=12, d=50, dense DeGroot, seed=0
- epochs=800, lr=0.005 (matched to successful rung-1 training)

## Primary metric (§5 — dense: relative Frobenius)
- **Attention recovery** ||A − W||_F / ||W||_F = **0.189623**
- Spearman (secondary) = 0.934304

## Ridge-baseline recovery (same data)
- Ridge rel-Frobenius = **0.200684**
- Ridge Spearman (from metrics path) — see `ridge_W.npy`

## Prediction MSE (held-out)
- Attention = **2.580179e-03**
- Ridge = 2.590654e-03

## State-covariance condition number
- κ(cov(X)) = **6.135020e+01**

## Heatmaps
`heatmap_A_vs_W.png`

## Verdict
Attention **matches or slightly beats** ridge on both recovery and prediction under process noise.
No failure-protocol trigger (attn 0.190 ≤ ridge 0.201).

Compared to rung 1 (σ=0): recovery degrades for both methods as expected (noisy targets),
but the attention–ridge gap closes — softmax row-stochasticity may act as helpful
regularization when the regression target is corrupted. Condition number remains healthy (~61).

## Proceed
Continue to rung 3 (FJ stubbornness + anchor tokens; keep σ=0.05).
