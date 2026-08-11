# Rung 0 Report — Ridge-regression baseline

## Setup
- N=50, M=200, T=12, sigma=0.0, dense DeGroot (Λ=I)
- Seed=0, ridge α=0.001
- Training pairs (held-in trajectories): 1920


## Primary metric (§5 — dense: relative Frobenius)
- **Ridge recovery** \(\|\hat{W} - W\|_F / \|W\|_F\) = **0.000006**
- Row-softmax-projected ridge rel-Frobenius = 0.000006
- Spearman (raw ridge vs W) = 1.000000

## Ridge-baseline recovery (this is the baseline)
Same as primary: 0.000006

## Prediction MSE (held-out trajectories)
- Held-out MSE with \(\hat{W}\): **8.231601e-14**
- Train MSE: 3.823212e-14

## State-covariance condition number
- κ(cov(X)) = **6.852416e+01**

## Artifacts
- `true_W.npy`, `true_Lambda.npy`, `ridge_W.npy`, `heatmap_A_vs_W.png`, `config.json`, `metrics.json`

## Verdict
PASS: data supports recovery (ridge rel-Frobenius < 0.1).

Gate for rung 1: ridge must show the data identifies W. Attention cannot beat missing information.
