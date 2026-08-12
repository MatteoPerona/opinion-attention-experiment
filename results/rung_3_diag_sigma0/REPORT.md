# Diagnostic: rung_3_diag_sigma0 — noiseless FJ, Adam, poor init

## Setup (actual run)
- **Diagnostic only** (not a ladder rung): isolate whether attention can recover ΛW when data is perfectly identifiable
- Dense FJ, N=50, M=200, T=12, d=50, **σ=0.0** (noiseless)
- Optimizer: **Adam** (lr=5e-3, 1000 epochs, batch=256)
- Init: **poor** — Normal(0, 0.02) ID embeddings + Xavier W_Q/W_K (pre-orthogonal-init era)
- Seed=[0]
- Value path: raw scalars; 2N anchor tokens; free attention

## Primary metric (§5 — dense: relative Frobenius on current block vs ΛW)
- **Attention** ||A_cur − ΛW||_F / ||ΛW||_F = **3.326411**
- Spearman = 0.580722
- Rescaled-to-W rel-Frobenius = 1.958964

## Ridge-baseline recovery (same data, N×2N operator)
- Ridge current-block vs ΛW = **0.000812**
- Ridge anchor-block vs (I−Λ) = 0.000119

## Stubbornness recovery (bonus)
- MAE of diag(A_anc) vs (1−λ) = **0.077090**
- Correlation = 0.557742

## Prediction MSE (held-out)
- Attention = **6.855458e-04**
- Ridge = 1.126017e-11

## State-covariance condition number
- κ = **3.462644e+01**

## Heatmaps
`heatmap_A_vs_W.png` (current block vs ΛW); per-seed also has anchor heatmap.

## Failure / gap diagnosis
Ridge recovers ΛW essentially perfectly under noiseless FJ (rel-F ≈ 8e-4), but Adam+poor-init attention fails badly (rel-F ≈ 3.33). This is an **architecture/optimization** failure, not a data failure — item 2 in `results/rung_3/REPORT.md` diagnostics table.
