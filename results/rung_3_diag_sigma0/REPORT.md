# Rung 3 Report — FJ + anchor tokens + noise

## Setup
- Exactly one change vs rung 2: **stubbornness on** (Λ≠I) with **2N anchor-token** design
- σ=0.0 retained; dense W; N=50, M=200, T=12, d=50
- Seeds=[0]

## Primary metric (§5 — dense: relative Frobenius on current block vs ΛW)
- **Attention** \(\|A_{cur} - \Lambda W\|_F / \|\Lambda W\|_F\) = **3.326411**
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
Attention current-block recovery worse than N×2N ridge operator. Possible causes: (1) optimization — 2N-way softmax harder; (2) architecture — Q/K from IDs must jointly represent ΛW and diagonal anchors; (3) capacity OK if d≥N. Consider `--extra-seeds 0,1,2`.

