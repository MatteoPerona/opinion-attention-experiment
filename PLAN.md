# Experiment Plan — Attention Recovers Opinion Dynamics

Restatement of `opinion-attention-experiment.md` (draft v0.4) before any code.
Where this plan and that document conflict, the document wins, except for the autonomy override in the run prompt (checkpoint protocol instead of hard stop after rung 1).

---

## Core question

Given only observed opinion trajectories from a known Friedkin–Johnsen (FJ) / DeGroot system, can a **single attention head** recover the hidden influence matrix \(W\) (and, under FJ, stubbornness via anchor tokens), while also predicting next states?

Success requires both: (1) low prediction MSE, and (2) learned attention map \(A\) matching true \(W\) (the prize).

---

## Generative model (ground truth)

**FJ update** (process noise for most rungs):

\[
\mathbf{x}(t+1) = \Lambda W \mathbf{x}(t) + (I-\Lambda)\mathbf{x}(0) + \boldsymbol{\varepsilon}(t)
\]

- \(W\): row-stochastic influence matrix (recover this)
- \(\Lambda = \mathrm{diag}(\lambda_i)\): susceptibility; \(\Lambda=I\) → DeGroot
- Steady-state closed form \(\mathbf{x}^* = (I-\Lambda W)^{-1}(I-\Lambda)\mathbf{x}(0)\) is **valid only when all \(\lambda_i < 1\)**; never use it for DeGroot labels — use empirical end-of-trajectory instead

**Dataset defaults (locked):** \(N=50\), \(T=10\)–\(15\), \(M=200\) trajectories from one fixed \((W,\Lambda)\) with varied \(\mathbf{x}(0)\). Dense random \(W\) for rungs 0–3; clustered/sparse at rung 4. Adaptive: if ridge recovery is poor, raise \(M\) first.

---

## Experimental ladder (execute 0→4; 5 only if clean; never 6–7)

| Rung | Change (exactly one vs prior) | Primary metric (§5) |
|------|-------------------------------|---------------------|
| **0** | Ridge OLS/ridge fit of \(W\) on \((\mathbf{x}(t),\mathbf{x}(t+1))\) — no neural net | Rel. Frobenius \(\| \hat W - W \|_F / \|W\|_F\) |
| **1** | DeGroot, dense \(W\), **no noise**, single head, many ICs | Rel. Frobenius \(\|A-W\|_F/\|W\|_F\); Spearman secondary |
| **2** | + process noise \(\sigma>0\) | Same (dense) |
| **3** | + FJ stubbornness + **anchor tokens** (\(2N\) tokens) | Same + stubbornness recovery (anchor weight vs \(1-\lambda_i\)) |
| **4** | Vary graph: sparse / clustered | **Spearman + top-\(k\) edge precision**; Frobenius secondary |
| **5** | Multiple heads (stretch only) | Per-head / aggregate recovery |
| **6–7** | Many-worlds / real data | **OUT OF SCOPE** — do not attempt |

Between rungs, change exactly one thing (§6). Re-run ridge baseline on each rung's data.

---

## Hard constraints (load-bearing)

1. **Identity embeddings drive Q/K** — attention over *who*, not opinion value. Learnable per-agent IDs; single fixed world per rung.
2. **Value path = raw scalar opinion** — no learned \(W_V\), no output projection, no MLP after attention. Prediction needing more capacity is a finding, not a license to add capacity.
3. **FJ anchoring (rung 3+)** = \(2N\) tokens (current + anchors); raw-scalar values for both. Agent \(i\)'s row should put \(\lambda_i W_{ij}\) on currents and \(1-\lambda_i\) on own anchor.
4. **Token axis = agents** — shape `[N, features]` (or `[2N, …]` under FJ). Time is only the training-pairs axis.
5. **Attention fully free** — never mask to true graph.
6. **Embedding dim \(d \geq N = 50\)**.
7. **One fixed world, \(M=200\) short trajectories**; raise \(M\) if ridge fails before touching the model.
8. **One-step teacher-forced training** — input true \(\mathbf{x}(t)\), predict \(\mathbf{x}(t+1)\); MSE loss.
9. **Do not silently violate constraints** — object in `REPORT.md`, then comply.

---

## Checkpoint protocol (this run's autonomy override)

After each rung \(k\), write `results/rung_<k>/REPORT.md` with:

- (a) primary metric for that rung (§5)
- (b) ridge-baseline recovery on the same data
- (c) prediction MSE on held-out trajectories
- (d) state-covariance condition number
- (e) side-by-side heatmaps of learned \(A\) vs true \(W\) (PNG)

Only then start the next rung.

**Failure protocol:** if attention recovery ≪ ridge on same data → diagnose (3 seeds, vs ridge), classify as data / capacity / optimization / architecture, write diagnosis into report. If unfixed after reasonable effort, document and **continue** with caveat. Documented failure is valid; silent patches are not.

---

## Scientific integrity

- Every number traces to a saved artifact; no fabrication.
- Primary metrics fixed before seeing results.
- Fixed seeds; config JSON; save true \(W\), learned \(A\), \(\Lambda\) as `.npy`.
- Negative results get equal detail.

---

## Engineering layout

```
sim/           # FJ/DeGroot generator
model/         # single attention head (identity Q/K, raw V)
baselines/     # ridge W recovery
experiments/   # one script per rung
results/       # rung_k/{config.json, *.npy, heatmaps, REPORT.md}
tests/         # generator unit tests (before rung 0)
```

Stack: Python, NumPy + PyTorch, CPU. argparse/JSON only. Commit after each rung with rung number in the message.

**Pre-rung-0 tests:** (1) rows of \(W\) sum to 1; (2) \(\Lambda=I\), \(\sigma=0\) → trajectory \(= W^t \mathbf{x}(0)\); (3) all \(\lambda_i<1\) → empirical steady state matches closed form; (4) closed form **not** used when \(\Lambda=I\).

---

## Final deliverable

`RESEARCH_REPORT.md`: table of primary metrics (attention vs ridge) per rung, inline heatmaps, what worked / failed and why, Limitations and open questions — written for skeptical reproduction.

---

## Execution order

1. Unit tests for `sim/`
2. Rung 0 → report → commit
3. Rung 1 → report → commit
4. Rung 2 → report → commit
5. Rung 3 → report → commit
6. Rung 4 → report → commit
7. Rung 5 only if 0–4 clean
8. Top-level `RESEARCH_REPORT.md`
