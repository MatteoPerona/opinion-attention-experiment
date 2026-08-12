# Diagnostic: rung_3_diag_lbfgs_s0 — noiseless FJ, LBFGS + orthogonal init

## Setup (actual run)
- **Diagnostic only**: same noiseless data regime as `rung_3_diag_sigma0`, better optimizer/init
- Dense FJ, N=50, M=200, T=12, d=50, **σ=0.0**
- Optimizer: **LBFGS** (lr=1.0, 50 outer steps, strong Wolfe)
- Init: **orthogonal** ID embeddings (×0.5) + identity W_Q/W_K
- Seed=[0]
- Value path: raw scalars; 2N anchor tokens; free attention

## Primary metric (§5 — dense: relative Frobenius on current block vs ΛW)
- **Attention** ||A_cur − ΛW||_F / ||ΛW||_F = **0.528092**
- Spearman = 0.553552
- Rescaled-to-W rel-Frobenius = 0.571965

## Ridge-baseline recovery (same data, N×2N operator)
- Ridge current-block vs ΛW = **0.000812**
- Ridge anchor-block vs (I−Λ) = 0.000119

## Stubbornness recovery (bonus)
- MAE of diag(A_anc) vs (1−λ) = **0.002945**
- Correlation = 0.999892

## Prediction MSE (held-out)
- Attention = **7.448823e-05**
- Ridge = 1.126017e-11

## State-covariance condition number
- κ = **3.462644e+01**

## Heatmaps
`heatmap_A_vs_W.png` (current block vs ΛW); per-seed also has anchor heatmap.

## Failure / gap diagnosis
Orthogonal init + LBFGS recovers **anchors** almost perfectly (stub corr ≈ 0.9999) and cuts ΛW rel-F from ~3.3 → ~0.53, but the current block still lags ridge (8e-4) by orders of magnitude. Anchors easy; dense ΛW under joint 2N-softmax still hard — item 3 in `results/rung_3/REPORT.md` diagnostics table.
