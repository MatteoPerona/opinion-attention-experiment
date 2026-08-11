# Rung 3 Report — FJ + anchor tokens + noise

## Setup
- Exactly one change vs rung 2: **stubbornness on** (Λ ≠ I) with **2N anchor-token** design (§3b note 3)
- σ = 0.05 retained from rung 2; dense W; N=50, d=50
- **M raised 200 → 2000** per §2 adaptive rule after ridge recovery of ΛW was poor at M=200
- Seeds = [0, 1, 2]; Adam lr=5e-3, 800 epochs; orthogonal ID init + identity W_Q/W_K
- Value path: raw scalars for current and anchor tokens (no W_V / out-proj / MLP)

## Primary metric (§5 — dense: relative Frobenius on current block vs ΛW)
- **Attention** ||A_cur − ΛW||_F / ||ΛW||_F = **0.407289** (seed 0)
- Spearman = 0.764527
- Rescaled-to-W rel-Frobenius = 0.519804

## Ridge-baseline recovery (same data, N×2N operator)
- Ridge current-block vs ΛW = **0.236520**
- Ridge anchor-block vs (I−Λ) = 0.028949

## Stubbornness recovery (bonus — cleanly measurable via anchor diagonal)
- MAE of diag(A_anc) vs (1−λ) = **0.006458**
- Correlation = **0.999528**

## Prediction MSE (held-out)
- Attention = **2.572250e-03**
- Ridge = 2.488466e-03

## State-covariance condition number
- κ(cov(X)) = **1.388260e+01**

## Heatmaps
`heatmap_A_vs_W.png` (current block vs ΛW); per-seed dirs also have `heatmap_anchor.png`.

## Multi-seed summary

| seed | attn rel-F vs ΛW | ridge rel-F | Spearman | stub MAE | stub corr | val MSE |
|------|------------------|-------------|----------|----------|-----------|---------|
| 0 | 0.407 | 0.237 | 0.765 | 0.0065 | 0.9995 | 2.57e-3 |
| 1 | 0.370 | 0.222 | 0.810 | 0.0057 | 0.9995 | 2.58e-3 |
| 2 | 0.421 | 0.214 | 0.747 | 0.0054 | 0.9993 | 2.57e-3 |

## Failure protocol / diagnosis

### What failed
Attention **current-block** recovery remains substantially worse than the unconstrained N×2N ridge operator (≈0.40 vs ≈0.23 rel-Frobenius). Softmax attention does **not** match ridge’s recovery of ΛW.

### What worked
**Stubbornness recovery succeeds**: diagonal anchor weight tracks (1−λ) with corr ≈ 0.999 and MAE ≈ 0.006. Prediction MSE matches ridge. The architecture’s claimed readout for stubbornness is empirically valid.

### Ablations / diagnostics (saved under `results/rung_3_diag_*`)
1. **M=200, σ=0.05 (initial):** ridge rel-F(cur)=0.83 — data under-exciting for ΛW → raised M (adaptive rule).
2. **M=200, σ=0 (noiseless):** ridge rel-F≈8e-4 (perfect); Adam+poor init → attn rel-F≈3.3 (architecture/opt fail even when data is fine).
3. **LBFGS + orthogonal init, σ=0:** attn rel-F≈0.53; stub corr≈0.999 — anchors easy, ΛW block still hard under softmax.
4. **LBFGS on σ=0.05, M=2000:** collapsed (pred MSE 0.28) — not a fix for the noisy case.
5. **Best noisy run:** Adam + orthogonal init + M=2000 (this report).

### Classification
- **Data (partial):** at M=200, σ=0.05, even ridge cannot recover ΛW (rel-F 0.83). Raising M fixes ridge to ~0.23; joint feature cov of [x(t); x(0)] is ill-conditioned (κ_Z ~10³) because FJ keeps x near x0.
- **Architecture / optimization (primary residual):** with identifiable data (σ=0 or large M), softmax over 2N ID-keyed tokens recovers anchors but only partially recovers the dense ΛW block. Capacity d=N is sufficient in principle (rank ≤ N covers the N×2N target of rank ≤ N). Gap is the constrained bilinear+softmax parameterization + nonconvex training vs unconstrained ridge — **not** missing W_V/MLP (those remain forbidden; prediction MSE is already matched).

### Objection logged (constraint compliance)
If prediction-only were the goal, a learned value mix of x and x0 could help — but §3b forbids it, and prediction is already near-ridge. The open gap is **interpretability of A_cur as ΛW**, not next-step MSE. Constraint respected.

## Verdict
**Partial success / documented failure on the prize metric.** Anchor-token stubbornness readout works; current-block matrix recovery lags ridge and is the open problem carried into rung 4.

## Proceed
Continue to rung 4 (clustered graph; keep FJ+anchors+σ; switch primary metrics to Spearman + top-k). Carry caveat: A_cur ↔ ΛW recovery is imperfect.
