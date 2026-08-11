# Rung 1 Report — DeGroot dense, no noise, single head

## Setup
- N=50, M=200, T=12, d=50, σ=0, dense DeGroot
- Seeds=[0, 1, 2], epochs=800, lr=0.005, batch=256
- Identity-driven Q/K, raw-scalar V, fully free attention (no graph mask)

## Primary metric (§5 — dense: relative Frobenius)
- **Attention recovery** ||A − W||_F / ||W||_F = **0.024551** (seed 0)
- Spearman (secondary) = 0.998995

## Ridge-baseline recovery (same data)
- Ridge rel-Frobenius = **0.000006**
- Ridge Spearman = 1.000000

## Prediction MSE (held-out trajectories)
- Attention MSE = **1.320051e-06**
- Ridge MSE = 8.231601e-14

## State-covariance condition number
- κ(cov(X)) = **6.852416e+01**

## Heatmaps
See `heatmap_A_vs_W.png` (learned attention vs true W).

## Multi-seed diagnosis (failure protocol)
Initial single-seed run (400 epochs, lr=0.01) gave attn rel-F=0.059 vs ridge ~0 → triggered 3-seed re-run with longer training.

| seed | attn rel-F | ridge rel-F | attn Spearman | val MSE |
|------|------------|-------------|---------------|---------|
| 0 | 0.02455 | 5.9e-6 | 0.9990 | 1.32e-6 |
| 1 | 0.02622 | 5.6e-6 | 0.9990 | 1.45e-6 |
| 2 | 0.02268 | 5.4e-6 | 0.9991 | 1.09e-6 |

**Classification of residual gap (attn ~0.025 vs ridge ~0):**
- **Not data:** ridge recovers W essentially perfectly; κ≈50–70 is well-conditioned.
- **Not capacity:** d=N=50; logits = E M Eᵀ can represent any N×N matrix when E is full rank, and softmax(log W) realizes any positive row-stochastic W.
- **Optimization / architecture constraint:** Adam on MSE with softmax-parameterized A does not reach the unconstrained OLS optimum. Residual ~2–3% relative Frobenius is stable across seeds (not a seed lottery). Prediction MSE is already ~1e-6.

**Verdict for the research question:** structure recovery **succeeds** (Spearman ≈ 0.999, rel-F ≈ 0.025). Attention does not match ridge’s numerical ceiling; that gap is expected under the constrained parameterization and is reported, not patched by adding W_V / MLP capacity.

## Proceed
Continue to rung 2 (add process noise only).
