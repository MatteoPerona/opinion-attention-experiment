# Diagnostic: rung_3_diag_lbfgs_noise — noisy FJ, LBFGS collapsed

## Setup (actual run)
- **Diagnostic only**: attempt LBFGS on the best noisy rung-3 data regime
- Dense FJ, N=50, **M=2000**, T=12, d=50, **σ=0.05**
- Optimizer: **LBFGS** (lr=0.5, 30 outer steps) — **collapsed**
- Init: orthogonal ID embeddings (×0.5) + identity W_Q/W_K
- Seed=[0]
- Value path: raw scalars; 2N anchor tokens; free attention

## Primary metric (§5 — dense: relative Frobenius on current block vs ΛW)
- **Attention** ||A_cur − ΛW||_F / ||ΛW||_F = **0.611646**
- Spearman = 0.016295
- Rescaled-to-W rel-Frobenius = 0.503433

## Ridge-baseline recovery (same data, N×2N operator)
- Ridge current-block vs ΛW = **0.236520**
- Ridge anchor-block vs (I−Λ) = 0.028949

## Stubbornness recovery (bonus)
- MAE of diag(A_anc) vs (1−λ) = **0.477387**
- Correlation = 0.231623

## Prediction MSE (held-out)
- Attention = **2.818350e-01**
- Ridge = 2.488466e-03

## State-covariance condition number
- κ = **1.388260e+01**

## Heatmaps
`heatmap_A_vs_W.png` (current block vs ΛW); per-seed also has anchor heatmap.

## Failure / gap diagnosis
LBFGS on σ=0.05 / M=2000 **collapsed** (pred MSE 0.28 ≫ ridge 2.5e-3; stub corr ≈ 0.23). Not a fix for the noisy case — item 4 in `results/rung_3/REPORT.md` diagnostics table. Best noisy run remains Adam + orthogonal init + M=2000 in the main rung-3 report.
