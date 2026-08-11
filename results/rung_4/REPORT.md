# Rung 4 Report — clustered graph + FJ + noise

## Setup
- Exactly one change vs rung 3: **graph kind = clustered** (5 clusters, strong within / weak between)
- FJ + anchor tokens + σ=0.05 retained; N=50, M=2000, T=12, d=50
- Seeds=[0,1,2]; Adam lr=5e-3, 800 epochs; orthogonal ID init
- Primary metrics switched per §5: **Spearman + top-k edge precision** (Frobenius secondary; positivity floor)

## Primary metrics (§5 — sparse/clustered)
- **Spearman (A_cur vs W)** = **0.602254** (seed 0)
- **Top-k precision (A_cur vs W, k=#true edges)** = **0.807921**
- Spearman vs ΛW = 0.604803
- Top-k vs ΛW = 0.807921
- Rel-Frobenius vs ΛW (secondary) = 0.173127

## Ridge-baseline recovery (same data)
- Ridge Spearman (current vs ΛW) = **0.624981**
- Ridge rel-Frobenius (current vs ΛW) = **0.101194**

## Stubbornness recovery
- MAE diag(A_anc) vs (1−λ) = 0.007882

## Prediction MSE (held-out)
- Attention = **2.582065e-03**
- Ridge = 2.484346e-03

## State-covariance condition number
- κ(cov(X)) = **1.361869e+01**

## Heatmaps
`heatmap_A_vs_W.png`

## Multi-seed summary

| seed | Spearman vs W | top-k vs W | ridge Spearman | stub MAE | val MSE |
|------|---------------|------------|----------------|----------|---------|
| 0 | 0.602 | 0.808 | 0.625 | 0.0079 | 2.58e-3 |
| 1 | 0.595 | 0.822 | 0.642 | 0.0222 | 2.66e-3 |
| 2 | 0.610 | 0.851 | 0.645 | 0.0093 | 2.57e-3 |

## Diagnosis
Attention **matches ridge on the named primary (Spearman)** within ~0.02–0.04 and achieves **top-k edge precision ≈ 0.81–0.85**.
No failure-protocol trigger (attn Spearman not substantially below ridge).

Caveat carried from rung 3: Frobenius recovery of ΛW remains imperfect vs ridge (seed1 rel-F 0.58 is an outlier; seeds 0/2 ~0.17 vs ridge ~0.10). For clustered graphs this is expected to matter less — §5 explicitly prefers rank/top-k because softmax positivity floors Frobenius on sparse support.

Within-cluster collinearity (§2) limits both methods’ Spearman to ~0.60–0.64 even for ridge — individual edge weights are partially unidentifiable; structure (which edges exist) is still recoverable at high top-k precision.

## Verdict
**Pass on primary metrics** for the clustered stress test: attention discovers the influence skeleton competitively with ridge. Rung-3 caveat on dense ΛW Frobenius recovery still applies but does not dominate the sparse primary.

## Stretch rung 5
Not attempted: rung 3’s dense ΛW recovery gap means 0–4 did not all succeed cleanly; multi-head would muddy interpretability further (§6).
