# Task 3 Report — FJ σ-sweep (rung-3 configuration)

## Setup
- Dense FJ + 2N anchor tokens; N=50, M=2000, T=12, d=50
- Adam lr=0.005, epochs=800, orthogonal ID init
- Vary **only σ** ∈ [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
- Seed 0 at every σ; seeds {0,1,2} at σ ∈ {0.05, 0.2}

## Hypotheses (pre-registered before results)
1. κ_Z falls as σ rises (noise decorrelates x(t) from x(0))
2. ridge ΛW recovery is non-monotone in σ with an interior optimum (§1b)
3. attention−ridge gap narrows as σ rises (flat current-vs-anchor dirs re-excited)

## Primary metric (§5 — dense: rel-Frobenius A_cur vs ΛW) — seed 0

| σ | attn | raw ridge | proj ridge | Spearman | stub corr | attn MSE | κ | κ_Z | attn−ridge gap |
|---|------|-----------|------------|----------|-----------|----------|---|-----|----------------|
| 0 | 0.2998 | 0.0001 | 0.0001 | 0.8421 | 1.0000 | 8.7072e-06 | 1.550e+01 | 4.037e+03 | 0.2997 |
| 0.02 | 0.3734 | 0.1117 | 0.1069 | 0.7870 | 0.9998 | 4.2969e-04 | 1.418e+01 | 2.932e+03 | 0.2618 |
| 0.05 | 0.4073 | 0.2365 | 0.2190 | 0.7645 | 0.9995 | 2.5722e-03 | 1.388e+01 | 1.262e+03 | 0.1708 |
| 0.1 | 0.4246 | 0.3633 | 0.3236 | 0.7336 | 0.9989 | 1.0152e-02 | 1.294e+01 | 4.189e+02 | 0.0612 |
| 0.2 | 0.4317 | 0.4752 | 0.4067 | 0.7150 | 0.9976 | 4.0336e-02 | 1.026e+01 | 1.161e+02 | -0.0435 |
| 0.35 | 0.4343 | 0.5317 | 0.4453 | 0.7052 | 0.9961 | 1.2300e-01 | 6.796e+00 | 4.047e+01 | -0.0974 |
| 0.5 | 0.4405 | 0.5516 | 0.4586 | 0.7125 | 0.9947 | 2.5068e-01 | 4.799e+00 | 2.133e+01 | -0.1111 |

## Multi-seed checks
- σ=0.05: attn rel-F=[0.4073, 0.3702, 0.4213]; ridge=[0.2365, 0.2224, 0.2144]; κ_Z=[1261.8, 1128.4, 1156.7]
- σ=0.2: attn rel-F=[0.4317, 0.4046, 0.4528]; ridge=[0.4752, 0.4468, 0.4633]; κ_Z=[116.1, 110.4, 108.5]

## Ridge-baseline recovery
See raw / projected columns above (same data per σ).

## Prediction MSE
See attn MSE column; ridge MSE in `metrics.json` per run.

## State-covariance condition numbers
- Marginal κ and joint κ_Z both logged; plot: `recovery_and_kappa_vs_sigma.png`

## Heatmaps
Per-σ / per-seed under `sigma_*/seed_*/heatmap_A_vs_W.png`

## Verdicts on hypotheses

### (i) κ_Z falls as σ rises — **SUPPORTED**
Spearman(σ, κ_Z) = −1.0. Joint condition number drops from ~4037 at σ=0 to ~21 at σ=0.5 while marginal κ only falls from ~15 to ~5. Run #1’s marginal κ≈14 hid the real N×2N pathology.

### (ii) ridge ΛW recovery non-monotone with interior optimum — **NOT SUPPORTED**
With M=2000 dense FJ, ridge is **best at σ=0** (rel-F≈6e-5) and degrades monotonically as σ rises. No interior optimum in this grid. §1b’s “some noise helps” does not appear for this well-excited multi-trajectory FJ operator fit — noise here is mostly target corruption. (Negative result retained; not re-tuned.)

### (iii) attention−ridge gap narrows as σ rises — **SUPPORTED**
Spearman(σ, attn−ridge gap) = −1.0. Gap falls from +0.30 at σ=0 to **negative** by σ=0.2 (attention beats raw ridge). Consistent with flat current-vs-anchor directions getting re-excited, and/or softmax acting as helpful regularization when targets are noisy. Caveat: absolute attention recovery still plateaus ~0.43–0.44; the gap closes mainly because ridge worsens faster.

## Plot
`recovery_and_kappa_vs_sigma.png`
