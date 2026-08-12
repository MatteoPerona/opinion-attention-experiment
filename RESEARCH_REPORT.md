# Research Report: Can Attention Recover Opinion Dynamics?

**Experiment design:** `opinion-attention-experiment.md` (draft v0.4)  
**Execution plan:** `PLAN.md`  
**Scope executed:** rungs 0–4 (rung 5 deferred; rungs 6–7 out of scope)  
**Stack:** NumPy + PyTorch (CPU), N=50, single fixed world per rung, one-step teacher-forced MSE  

All numbers below are taken from `results/rung_*/metrics.json` (primary seed 0 unless noted). Re-run scripts live in `experiments/`.

---

## Summary table — primary metrics vs ridge

| Rung | Setting | Primary metric | Attention | Ridge (same data) | Held-out pred MSE (attn / ridge) | κ(cov X) |
|------|---------|----------------|-----------|-------------------|----------------------------------|----------|
| **0** | Dense DeGroot, σ=0, ridge only | rel-Frobenius vs W | — | **5.92e-6** | 8.23e-14 / — | 68.5 |
| **1** | Dense DeGroot, σ=0, single head | rel-Frobenius vs W | **0.0246** | 5.92e-6 | 1.32e-6 / 8.23e-14 | 68.5 |
| **2** | + process noise σ=0.05 | rel-Frobenius vs W | **0.190** | 0.201 | 2.58e-3 / 2.59e-3 | 61.4 |
| **3** | + FJ + 2N anchors (M=2000) | rel-Frobenius vs ΛW | **0.407** | 0.237 | 2.57e-3 / 2.49e-3 | 13.9 |
| **4** | + clustered W | Spearman vs W / top-k | **0.602 / 0.808** | Spearman(ΛW) 0.625 | 2.58e-3 / 2.48e-3 | 13.6 |

Secondary (rung 3 stubbornness): diag(A_anc) vs (1−λ) — MAE **0.00646**, corr **0.9995**.

---

## Heatmaps (learned map vs ground truth)

### Rung 0 — ridge Ŵ vs true W
![rung0](results/rung_0/heatmap_A_vs_W.png)

### Rung 1 — attention A vs true W (DeGroot, noiseless)
![rung1](results/rung_1/heatmap_A_vs_W.png)

### Rung 2 — attention A vs true W (DeGroot + noise)
![rung2](results/rung_2/heatmap_A_vs_W.png)

### Rung 3 — attention current block vs ΛW (FJ + anchors)
![rung3](results/rung_3/heatmap_A_vs_W.png)

### Rung 4 — attention current block vs clustered W
![rung4](results/rung_4/heatmap_A_vs_W.png)

---

## What worked

1. **Data gate (rung 0).** With M=200 short trajectories from varied x(0), dense noiseless DeGroot is fully identifiable: ridge rel-Frobenius ≈ 6×10⁻⁶. Multi-trajectory excitation (§1b) is load-bearing; the generator unit tests pass (row-stochastic W; Wᵗ dynamics; FJ closed form only when λᵢ<1).

2. **DeGroot attention recovers structure (rung 1).** Identity-driven Q/K + raw-scalar V recovers W with Spearman ≈ 0.999 and rel-Frobenius ≈ 0.025 across 3 seeds. The interpretability claim holds in the cleanest setting.

3. **Noise does not break DeGroot recovery (rung 2).** At σ=0.05, attention **matches or slightly beats** ridge (0.190 vs 0.201) with tied prediction MSE. Softmax row-stochasticity behaves like helpful regularization under noisy targets.

4. **Anchor-token stubbornness readout (rung 3).** diag(A_anc) recovers (1−λ) with correlation ≈ 0.999. The architectural rhyme between FJ anchors and attention tokens is empirically real.

5. **Clustered structure discovery (rung 4).** On the §5 primary metrics, attention Spearman ≈ 0.60 ≈ ridge 0.62 and top-k edge precision ≈ 0.81–0.85. Influence *skeleton* is recoverable even when dense entrywise Frobenius is messy.

---

## What failed (and why we think it failed)

### Rung 1 residual vs ridge (mild)
Attention never reaches ridge’s numerical ceiling (0.025 vs 6×10⁻⁶) under noiseless DeGroot.  
**Class:** optimization under softmax / bilinear ID parameterization — **not** data (ridge perfect) or capacity (d=N suffices). Prediction MSE already ~10⁻⁶. Not patched by adding W_V/MLP (forbidden; would muddle the prize metric).

### Rung 3 current-block recovery (main failure)
Attention ||A_cur − ΛW||_F / ||ΛW||_F ≈ **0.41** vs ridge **0.24** (M=2000, σ=0.05, 3 seeds).  
**Class:** architecture + optimization, with a data contribution at small M:

| Diagnostic | Ridge cur | Attn cur | Note |
|------------|-----------|----------|------|
| M=200, σ=0.05 | 0.83 | ~3.7 | data under-exciting → raised M |
| M=200, σ=0 | 8×10⁻⁴ | ~3.3 (bad init) / 0.53 (LBFGS+orth) | architecture/opt even when data is fine |
| M=2000, σ=0.05, Adam+orth | 0.24 | **0.41** | best noisy run; stub corr≈0.999 |

Anchors are easy; the dense ΛW block under a joint 2N-way softmax is hard. Joint [x(t); x(0)] covariance is ill-conditioned (κ_Z ~10³) because FJ keeps opinions near anchors — loss landscape is flatter for splitting mass between current and anchor tokens when their values are similar. Prediction MSE nonetheless matches ridge, so the model can predict without recovering A_cur ≈ ΛW — exactly the failure mode §3b warns about if the value path is unconstrained; here V is constrained and the failure is in the softmax factorization itself.

### Rung 5 not attempted
Rung 3’s prize-metric gap means 0–4 did not all succeed cleanly. Multi-head would further split influence across heads before the single-head FJ map is solved.

---

## Constraints compliance checklist

| Constraint | Status |
|------------|--------|
| Identity embeddings drive Q/K | Yes |
| Raw-scalar V; no W_V / out-proj / MLP | Yes |
| FJ = 2N anchor tokens, raw values | Yes (rung 3+) |
| Token axis = agents | Yes |
| Attention fully free (no graph mask) | Yes |
| d ≥ N = 50 | Yes |
| One world, M trajectories; raise M if ridge poor | Yes (M→2000 at rung 3) |
| One change between rungs | Yes |
| Never rungs 6–7 | Yes |

---

## Run #2 — corrections, σ-sweep, unrolling, factored diagnostic

Bookkeeping fixes + tests of the flat-landscape hypothesis. Primary metrics and success criteria fixed before seeing results. All run-#1 hard constraints retained except Task 5 (labeled ablation).

### Summary table

| Task | Setting | Primary | Attention | Raw ridge | Proj ridge | Notes |
|------|---------|---------|-----------|-----------|------------|-------|
| **1** | Hygiene | — | — | — | — | Diag configs fixed; Euclidean simplex proj + κ_Z helpers + tests |
| **2** | Rung 2 redo, 3 seeds, σ=0.05 | rel-F vs W | **0.1877** (mean) | 0.1961 | **0.1901** | Claim vs *projected* ridge **CONFIRMED** |
| **3** | FJ σ-sweep, M=2000 (seed0 @ σ=0.05) | rel-F A_cur vs ΛW | 0.407 | 0.237 | 0.219 | See sweep plot; κ_Z logged |
| **4** | Unrolled k=1→2→4, σ=0.05 | rel-F A_cur vs ΛW | **0.409** (mean) | ~0.22–0.24 | ~0.20–0.22 | **NO SUCCESS** vs one-step 0.40 |
| **5** | Factored-logit **diag** (not headline) | rel-F A_cur vs ΛW | **0.094** | 0.237 | 0.219 | Joint 2N-softmax was the bottleneck |

Task 2 per-seed (rel-F): attn [0.190, 0.189, 0.184]; raw ridge [0.201, 0.196, 0.192]; proj ridge [0.195, 0.190, 0.186].

### σ-sweep plot

![sigma sweep](results/rung_3_sigma_sweep/recovery_and_kappa_vs_sigma.png)

Seed-0 table (rel-F A_cur vs ΛW / κ_Z):

| σ | attn | raw ridge | proj ridge | κ_Z | attn−ridge gap |
|---|------|-----------|------------|-----|----------------|
| 0 | 0.300 | 0.000 | 0.000 | 4037 | +0.300 |
| 0.02 | 0.373 | 0.112 | 0.107 | 2932 | +0.262 |
| 0.05 | 0.407 | 0.237 | 0.219 | 1262 | +0.171 |
| 0.1 | 0.425 | 0.363 | 0.324 | 419 | +0.061 |
| 0.2 | 0.432 | 0.475 | 0.407 | 116 | **−0.044** |
| 0.35 | 0.434 | 0.532 | 0.445 | 40 | −0.097 |
| 0.5 | 0.441 | 0.552 | 0.459 | 21 | −0.111 |

### Verdicts on σ-sweep hypotheses (pre-registered)

1. **κ_Z falls as σ rises — SUPPORTED.** Spearman(σ, κ_Z)=−1.0 (4037→21). Marginal κ alone (~14 in run #1) hid the N×2N pathology.
2. **Ridge ΛW recovery non-monotone with interior optimum — NOT SUPPORTED.** Best at σ=0; degrades monotonically at M=2000. “Some noise helps” (§1b) does not appear for this well-excited FJ operator fit — noise is mostly target corruption here. Negative result retained.
3. **Attention−ridge gap narrows as σ rises — SUPPORTED.** Spearman(σ, gap)=−1.0; gap turns negative by σ=0.2. Absolute attention recovery still plateaus ~0.43–0.44; gap closes mainly because ridge worsens faster. Consistent with re-excitation of flat current-vs-anchor directions and/or softmax regularization under noise.

### Unrolled-training success criterion

**Criterion (pre-registered):** mean unrolled rel-F improves on one-step ≈0.40 by > seed spread ≈0.05 (i.e. mean < 0.35); interesting if ≈ ridge 0.23; stub corr > 0.99.

**Result: NO SUCCESS.** Mean unrolled rel-F = **0.409** (seeds 0.424 / 0.391 / 0.414) — indistinguishable from one-step. Training stable (no NaN/explosion with curriculum + grad clip). Stubbornness corr stays ≈0.9998. Unrolling alone does not cure the joint-softmax / flat-landscape gap.

### Factored-logit diagnostic (Task 5 — ablation, not headline)

Factored gates + N-token softmax: rel-F(A_cur vs ΛW) = **0.094**, Spearman 0.987, gate↔(1−λ) corr ≈ 1.000 — **beats ridge (0.237)**. Interprets the run-#1/Task-4 failure: the gap was largely the *joint* 2N-way softmax coupling currents and anchors, not the bilinear ID map itself. Remains outside the locked run-#1 architecture; do not promote to headline without a new protocol decision.

### Updated limitations and open questions

1. **Joint 2N-softmax is the FJ bottleneck (sharpened).** Task 5 shows a factored gate recovers ΛW better than ridge under the same data/constraints on V. Open: is a factored (or similarly structured) head an acceptable “single attention head” for the research claim, or a different operator class?
2. **Unrolling does not fix joint softmax (§4 graduation path tested).** Task 4 negative: multi-step self-rollout preserves prediction and stubbornness but not A_cur quality.
3. **Noise helps the attention−ridge *gap*, not absolute recovery.** Absolute attn rel-F worsens mildly with σ; ridge worsens faster. Interior optimum for ridge recovery not observed at M=2000 dense FJ.
4. **Always log κ_Z for FJ.** Marginal κ is misleading; joint [x(t); x(0)] conditioning is the right diagnostic.
5. **Fair ridge comparisons need simplex projection.** Task 2 confirmed attention still matches/beats projected ridge under DeGroot noise; raw-only comparisons overstate attention’s edge.
6. **Clustered / many-worlds / real data** — unchanged deferrals from run #1; rungs 5–7 not attempted.

### Reproduce (Run #2)

```bash
PYTHONPATH=. pytest -q tests/
PYTHONPATH=. python experiments/rung_2_noise.py --extra-seeds 0,1,2
PYTHONPATH=. python experiments/rung_3_sigma_sweep.py
PYTHONPATH=. python experiments/rung_3_unrolled.py --extra-seeds 0,1,2
PYTHONPATH=. python experiments/rung_3_factored_diag.py  # diagnostic only
```

Artifacts: `results/rung_2/`, `results/rung_3_sigma_sweep/`, `results/rung_3_unrolled/`, `results/rung_3_factored_diag/`.

### Run #1 reproduce (unchanged)

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest -q tests/
PYTHONPATH=. python experiments/rung_0_ridge.py
PYTHONPATH=. python experiments/rung_1_degroot.py --extra-seeds 0,1,2 --epochs 800 --lr 5e-3
PYTHONPATH=. python experiments/rung_2_noise.py --extra-seeds 0,1,2
PYTHONPATH=. python experiments/rung_3_fj.py --M 2000 --epochs 800 --lr 5e-3 --sigma 0.05 --extra-seeds 0,1,2
PYTHONPATH=. python experiments/rung_4_clustered.py --M 2000 --epochs 800 --lr 5e-3 --extra-seeds 0,1,2
```

---

*Run #2 written for skeptical reproduction. Hypothesis (ii) and Task 4 are explicit negatives.*
