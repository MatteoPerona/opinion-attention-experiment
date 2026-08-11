# 🧠 Can Attention Recover Opinion Dynamics?

### An experiment design for learning influence structure with a single attention head

> **One-line thesis:** If a single attention head really behaves like an opinion-dynamics operator, then when we feed it opinion trajectories from a *known* Friedkin–Johnsen system, the attention map it learns should reconstruct the true influence matrix — and predict the steady state.

---

## 🎯 0. The core question

We want to answer one thing before anything fancy:

> **Given only the observed opinions over time, can a single attention head recover the hidden influence structure that generated them?**

This is the **sanity check that gates the whole research program**. If we can't recover structure in a synthetic world where we *know* the ground truth, there's no hope on messy Reddit/Twitter data later. Nail this first. 🔒

**Success looks like:**

1. 📈 The model predicts the next opinion step / steady state accurately.
2. 🗺️ The learned attention map **matches the true influence matrix** we hid.

Point 2 is the real prize — that's the interpretability claim.

---

## 🧩 1. The generative model (our ground truth)

We simulate opinions with the **Friedkin–Johnsen (FJ)** model, with plain **DeGroot as a special case**.

### The FJ update

$$
\mathbf{x}(t+1) = \Lambda\, W\, \mathbf{x}(t) + (I - \Lambda)\, \mathbf{x}(0)
$$

| Symbol            | Meaning                                                                         | Shape          |
| ----------------- | ------------------------------------------------------------------------------- | -------------- |
| $\mathbf{x}(t)$ | opinion vector at time$t$                                                     | $N \times 1$ |
| $W$             | **influence matrix** (row-stochastic) — *the thing we want to recover* | $N \times N$ |
| $\Lambda$       | diagonal**susceptibility** matrix, $\lambda_i \in [0,1]$                | $N \times N$ |
| $\mathbf{x}(0)$ | innate / anchored opinions                                                      | $N \times 1$ |

🔑 **Why FJ over plain DeGroot?** DeGroot collapses *everyone* to a single consensus — too clean, and real data never looks like that. The $(I-\Lambda)\mathbf{x}(0)$ term keeps each agent partly anchored to its original stance, which is what produces **persistent disagreement and clusters** — the texture of real opinion data.

🎁 **Bonus:** that stubbornness term maps neatly onto an **anchor token** in the attention layout (see §3b, note 3) — the architecture and the model rhyme.

### DeGroot as a free special case

Set $\Lambda = I$ (susceptibility = 1, stubbornness = 0) and FJ reduces to:

$$
\mathbf{x}(t+1) = W\,\mathbf{x}(t)
$$

So we get **both models from one generator** — we can test recovery on the easy DeGroot case *first*, then turn on stubbornness.

### 🌫️ Adding noise

Real people are noisy — influences are unclear and choices aren't optimal. Noise acts as regularization, and — crucially — as **persistent excitation**: it keeps kicking the state out of the shrinking subspace the dynamics would otherwise collapse into, so the data keeps probing new directions of $W$ (see ⚠️ §1b). Counterintuitively, *some* noise makes recovery **easier**, not harder.

**Process noise (start here** ✅**)** — compounds over time, matches messy human decision-making:

$$
\mathbf{x}(t+1) = \Lambda\, W\, \mathbf{x}(t) + (I - \Lambda)\, \mathbf{x}(0) + \boldsymbol{\varepsilon}(t), \quad \boldsymbol{\varepsilon}(t) \sim \mathcal{N}(0, \sigma^2 I)
$$

**Observation noise (later)** — true dynamics stay clean, only what the model *sees* is corrupted. Tests robustness of the readout rather than the dynamics.

---

## ⚠️ 1b. Identifiability — READ BEFORE DESIGNING THE DATA

**A single trajectory does not identify $W$.** A noiseless DeGroot run gives you $\mathbf{x}(0), W\mathbf{x}(0), W^2\mathbf{x}(0), \dots$ — a **Krylov subspace**. $W$ is only pinned down on the span of the states actually visited, and both DeGroot and FJ converge **geometrically**, so that span collapses toward one (or a few) directions almost immediately. From one clean trajectory you can recover $W$'s action on a handful of directions — never the full $N \times N$ matrix.

Two consequences, both baked into this design:

1. **🚿 Many trajectories, one world.** We keep a single fixed $(W, \Lambda)$ — this is still single-world system identification — but we generate **many trajectories from many random initial conditions $\mathbf{x}(0)$**. Same identity embeddings, same matrix to recover; the varied starts are what excite all of $W$'s directions. (§2, step 6.)
2. **✂️ Many short trajectories beat one long one.** Because convergence is geometric, nearly all the information lives in the **transient**. Past the first ~dozen steps, new time steps add almost nothing. Budget compute accordingly: lots of short runs, not one epic one.

> 🚨 **Corrected expectation for the ladder's first rung:** the clean, noiseless, single-trajectory case is the *least* identifiable setting, not the easiest. "Perfect recovery" is only expected once initial conditions (and/or noise) provide enough excitation. Rung 1 (§6) is defined with multi-trajectory data for exactly this reason.

---

## 🧩 1c. Rung 0 — the regression baseline (mandatory, before any attention)

Since opinions are scalar and agent identities are fixed, the attention head is ultimately a constrained parameterization of an $N \times N$ matrix. So before training *any* neural model:

**Fit $W$ directly by (ridge-regularized) least squares** on the stored $(\mathbf{x}(t), \mathbf{x}(t+1))$ pairs. Five lines of code.

- If OLS/ridge **recovers** $W$ → the data supports recovery; any failure of the attention head is an **architecture/optimization** problem.
- If OLS/ridge **fails** → the data itself is under-exciting; fix §1b before touching the model. No attention head can beat the information that isn't there.

This baseline also gives us the **recovery ceiling** to compare the attention head against on every subsequent rung. 📐

---

## 🏗️ 2. The synthetic dataset

**Step-by-step generation:**

1. 👥 Pick $N$ agents: **N = 50** for run #1.
2. 🎲 Sample innate opinions $\mathbf{x}(0)$ (e.g. uniform or Gaussian).
3. 🕸️ Sample a **row-stochastic** influence matrix $W$ (non-negative rows summing to 1). 📌 **Decision: DENSE RANDOM for run #1** (e.g. i.i.d. positive entries, rows normalized). Clustered graphs are more realistic — but they are also exactly where within-cluster collinearity makes individual entries unidentifiable (agents in a cluster converge to similar traces, so their columns correlate and edge weights get mushy). Starting clustered would muddy the very first recovery number with a data-side confound. Dense random is the setting where "recovery works / doesn't" is a clean verdict on the *architecture*. Clustered structure enters at **rung 4**, as the stress test it's meant to be.
4. 🎚️ Sample susceptibilities $\lambda_i$ (all = 1 for the DeGroot phase; varied in $[0,1]$ for FJ).
5. 🔁 Iterate the update for **T = 10–15 steps** per trajectory, injecting process noise, and **record the full trajectory** $\mathbf{x}(0), \dots, \mathbf{x}(T)$ plus the steady state $\mathbf{x}^\*$. (Short on purpose — past the transient, new steps add almost nothing; §1b.)
6. 🗃️ **One world, many trajectories: M = 200.** Fix a single $(W, \Lambda)$ and generate **$M = 200$ trajectories from 200 different random initial conditions**. That yields ~2,000–3,000 training pairs against the 2,500 unknowns in $W$ — enough for a well-conditioned ridge fit with varied starts, and cheap to generate. 📏 **Adaptive rule: if rung-0 ridge recovery is poor, increase $M$ (and re-check the state-covariance condition number) BEFORE touching the model.** This is still single-world **system identification** — one matrix, fixed identity embeddings — but with enough excitation that "learned attention map = true $W$" is actually achievable. Generating many *worlds* (fresh $W$ per sample) is a *different, harder* experiment — amortized inference — parked in the extensions.

**FJ steady state (for reference / labels):**

$$
\mathbf{x}^\* = (I - \Lambda W)^{-1}(I - \Lambda)\,\mathbf{x}(0)
$$

> ⚠️ **Validity note:** this closed form requires $\rho(\Lambda W) < 1$, which holds when every $\lambda_i < 1$ — but **fails at $\Lambda = I$** (the DeGroot phase), where $\rho(W) = 1$ and the matrix is singular. In the DeGroot phase there is no anchored steady state; the system drifts to consensus (whose value depends on $\mathbf{x}(0)$). **Do not use this formula for DeGroot labels** — use the empirical end-of-trajectory state if a "final state" target is needed.

---

## 🔬 3. The architecture — one head, one matrix

We deliberately start with a **single attention head**. 🎯

**Why single:**

- The synthetic world has exactly **one** true influence matrix → one thing to recover.
- A single head is the cleanest possible mirror of it.
- Multiple heads split influence across heads → interpretability gets muddy before we've proven anything.
- If it breaks, there's **nowhere for the bug to hide**.

**The mechanics (standard Q/K/V):**

- Each agent's opinion (+ identity embedding) → an embedding vector.
- Multiply by learnable $W_Q, W_K, W_V$ → query, key, value per node.
- Attention scores: $\text{score}(i,j) = \frac{q_i \cdot k_j}{\sqrt{d}}$, then softmax over $j$.
- New state = weighted sum of value vectors.

📐 **Capacity requirement — set $d \geq N$ for dense $W$.** The pre-softmax logit matrix $q_i \cdot k_j$ has rank $\leq d$. A generic dense $W$ needs rank up to $N$, so an embedding dimension of, say, 16 with $N=50$ **silently caps recovery at a low-rank approximation** no matter how long you train. For $N = 50$ this is cheap — just choose $d \geq 50$ deliberately, don't inherit a small default. 🎛️ Flip side: on the clustered rung, $W$ is approximately low-rank, so *reducing* $d$ becomes an interesting experimental knob (how small can $d$ go before recovery of the cluster structure degrades?).

🪞 **The recovery claim:** the learned, row-stochastic attention map $A$ should line up with the true $W$ (DeGroot) or the full FJ operator including the anchor weights (see §3b, note 3).

---

## ⚠️ 3b. Notes for the implementer — READ BEFORE CODING

A generic Q/K/V head will happily minimize prediction loss while recovering **nothing**. Four things must be wired deliberately, or the prize metric is silently meaningless:

1. **🆔 Give each agent a learnable identity embedding.** The opinion is a *scalar* — if queries/keys are functions of opinion value alone, two agents with the same opinion get identical attention, and you cannot represent a fixed, identity-indexed $W$. Each node needs a learnable ID vector so attention is over *who the node is*, not what it currently thinks. (This is why the first run is **single-world**: fixed IDs index a fixed $W$. They can't span worlds — see below.)
2. **➡️ Keep the value path near-identity on the opinion.** For $\mathbf{x}(t+1) = A\,\mathbf{x}(t)$ to line up with the true dynamics, the value must carry the raw opinion (values ≈ $x_j$), **not** a full learned projection into embedding space. If $W_V$ is unconstrained you'll get great prediction and an attention map that recovers nothing. Let $Q/K$ (identity-driven) shape *who attends to whom*; keep $V$ as the opinion itself.
3. **⚓ FJ's anchor term = anchor TOKENS, not a feature and not a self-loop.** The $(I-\Lambda)\mathbf{x}(0)$ term anchors to the *innate* opinion $x_i(0)$, which a softmax over *current* opinions cannot produce. **The DeGroot rung ($\Lambda=I$, no anchor) sidesteps this entirely — start there.** When stubbornness turns on, represent the innate opinions as **$N$ additional anchor tokens**: each forward pass has $2N$ tokens — $N$ current-opinion tokens and $N$ anchor tokens — with values = raw scalars for *both* (preserving note 2). Then agent $i$'s attention row over all $2N$ tokens should learn $\lambda_i W_{ij}$ on the current tokens and $(1-\lambda_i)$ on **its own anchor token** — and the full attention row is genuinely row-stochastic, which $\Lambda W$ alone is not. 🎁 Bonus: "stubbornness recovery" becomes exactly *the weight on the anchor token* — cleanly measurable, no diagonal-vs-anchor ambiguity. (Feeding $x_i(0)$ in as a per-node input *feature* is the fallback, but it forces a learned value transform and muddies note 2 — prefer anchor tokens.)
4. **🧱 Agents are the token axis, not time.** One forward pass = one time-slice $\mathbf{x}(t)$ unstacked into $N$ agent-tokens (plus $N$ anchor tokens in the FJ phase). Shape the input **`[N agents, features]`**, not `[1, N]` — the latter attends over a single token and recovers nothing. Time is the training-pairs axis (§4), never the attention axis.

> 🧭 **Single-world vs. many-worlds — don't let the agent pick silently.** Identity-based recovery is inherently *per-world* (world A's node 3 ≠ world B's node 3), so fixed ID embeddings can't span worlds. **First run = one fixed world (many trajectories from it — §1b), recover its $W$.** The many-worlds "learn to infer structure from any trajectory" version is *amortized inference* — a real and interesting experiment, but a different and harder one. It lives in the extensions, not run #1.

---

## 🏋️ 4. Training

**⚠️ Two different "one-steps" — don't confuse them:**

- **Simulation: run the FULL thing.** Generate entire trajectories $\mathbf{x}(0), \mathbf{x}(1), \dots, \mathbf{x}(T)$ with noise (see §2) — many of them, from varied $\mathbf{x}(0)$ (§1b). **Do not truncate the sim to one step.**
- **Training target (run #1): one-step-ahead, teacher-forced.** Feed the model a *real* stored state $\mathbf{x}(t)$ and have it predict $\mathbf{x}(t+1)$; score against the *true* $\mathbf{x}(t+1)$ from the stored trajectory. Every input is a ground-truth state — the model **never predicts off its own prediction** in this mode.

**Concretely:**

- **Input:** a real state $\mathbf{x}(t)$ from a stored (noisy) trajectory.
- **Target:** the true next state $\mathbf{x}(t+1)$ from that same trajectory.
- **Training pairs:** every consecutive pair across every trajectory → $M$ trajectories of $T$ steps yield **$M \times T$ pairs**.
- **Loss:** MSE between predicted and true $\mathbf{x}(t+1)$.
- **Weights are shared across all nodes** (like a conv filter across patches) — one influence rule, applied everywhere.

> 💬 *Open decision — the graduation path:* run #1 is **one-step, teacher-forced** (easiest clean signal). Later, **unroll to the steady state**: feed $\mathbf{x}(0)$, apply the learned operator repeatedly feeding the model its *own* output back in, and score the final $\mathbf{x}^\*$. That's the stronger dynamical claim but harder to train (errors compound, long gradients) — so it is *not* run #1.

---

## 📏 5. Evaluation — the money metrics

| Metric                                                                    | What it tells us                                                                    | 🎯                     |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------- |
| **Regression-baseline recovery** (ridge-fit $W$ vs. true $W$)   | Does the*data* support recovery at all? Sets the ceiling.                         | Gate 🚪                |
| **Prediction error** (MSE on held-out steps / trajectories)         | Can it model the dynamics at all?                                                   | Necessary              |
| **Matrix recovery** (distance between learned $A$ and true $W$) | *Did it learn the real structure?*                                                | **The prize** 🏆 |
| **Stubbornness recovery** (anchor-token weight vs. $1-\lambda_i$) | Did FJ's anchoring show up where the architecture says it should?                   | Bonus                  |
| **Noise robustness** (recovery vs. $\sigma$)                      | Where does it break? Expect an*interior optimum*, not monotone decay — see §1b. | Stress test            |

**Choosing the recovery metric honestly:** softmax attention weights are **strictly positive**, while a sparse $W$ has exact zeros — so Frobenius distance has a hard floor on sparse/clustered rungs and will *understate* recovery. Use a portfolio, but 📌 **name ONE primary metric per rung** so "did it work" is never left to vibes:

- **Dense rungs (0–3):** primary = **relative Frobenius error** $\|A - W\|_F / \|W\|_F$; report Spearman rank-correlation of entries as secondary.
- **Sparse/clustered rungs (4+):** primary = **Spearman rank-correlation** + **top-$k$ edge precision** ($k$ = true number of edges); Frobenius reported but understood to have a positivity floor.

📌 **Decision — attention is FULLY FREE, never masked to the true graph.** Masking to the known edge structure "to help training" makes the prize metric circular: you'd be recovering edge *weights* after being handed the edge *structure*. Discovering which edges exist is part of the test. (A masked variant may be run later as a *diagnostic* comparison only — never as the headline result.)

---

## 🪜 6. The experimental ladder

Climb one rung at a time — never change two things at once:

0. ⚪ **Ridge-regression baseline** (§1c) — no neural net. Confirms the data identifies $W$ and sets the recovery ceiling. **Re-run at every subsequent rung.**
1. 🟢 **DeGroot, dense random $W$, no noise, single head, many initial conditions** — the "does anything work at all" baseline. With enough varied $\mathbf{x}(0)$'s, near-perfect recovery expected. (⚠️ With a *single* trajectory it is **not** — that's an identifiability failure, not a model failure; see §1b.)
2. 🟡 **DeGroot + process noise** — noise now does double duty as excitation; recovery may *improve* at small $\sigma$ before degrading at large $\sigma$.
3. 🟠 **Friedkin–Johnsen (stubbornness on, anchor tokens) + noise** — the realistic target; check anchor-token weight = stubbornness.
4. 🔵 **Vary graph structure** (sparse, clustered) — hardest / most realistic recovery; switch primary metrics to rank-correlation / top-$k$ (§5); explore small-$d$ regime (§3).
5. 🟣 **Multiple heads** — the *reward* extension: e.g. heads as different topics / social contexts, each with its own influence structure.
6. 🌍 **Many-worlds / amortized inference** — instead of fixed ID embeddings, the model reads a trajectory and *infers* the structure of an unseen world. Different problem (learning an inference procedure, not one matrix), harder, and needs a different readout than identity attention. Do NOT conflate with run #1.
7. 🔴 **Real data** (Reddit → Twitter) — once the mining pipeline is ready and the synthetic case is solid.

> 🛑 **STOP CONDITION — mandatory checkpoint after rung 1.** Complete rung 0 and rung 1 only, then **stop and report** before proceeding: (a) ridge recovery number, (b) attention recovery number, (c) both against the primary metric for the rung (§5), (d) the state-covariance condition number, (e) prediction MSE. Do **not** continue to rung 2 without review. If rung-1 recovery is mediocre, the failure must be diagnosed (data? capacity? optimization? — run the ridge comparison and a few seeds) *before* building six more rungs on sand. Each subsequent rung likewise ends with a report against its primary metric before the next begins.

---

## 📌 7. Decisions locked for run #1

Formerly the open-questions list — now a record, so the coding agent inherits choices instead of making them silently:

| Question                       | Decision                                                                                                | Where       |
| ------------------------------ | ------------------------------------------------------------------------------------------------------- | ----------- |
| $N$, $T$, $M$            | **N = 50, T = 10–15, M = 200**, with the adaptive rule: poor ridge recovery → raise $M$ first | §2         |
| $W$ sampling                 | **Dense random for run #1**; clustered deferred to rung 4 (collinearity confound)                 | §2, step 3 |
| Recovery metrics               | Portfolio with**one named primary per rung** (rel. Frobenius dense; Spearman + top-$k$ sparse)  | §5         |
| Masked vs. free attention      | **Fully free** — masking makes the prize circular                                                | §5         |
| Small-$d$ on clustered $W$ | Genuinely open — an exploration knob for**rung 4**, decided then, not now                        | §3, §6    |
| Training target                | One-step teacher-forced for run#1; unrolling is the graduation path                                     | §4         |
| Agent autonomy                 | **Stops after rung 1** for review; reports per rung thereafter                                    | §6         |

---

*Draft v0.4 — a living document, but no longer an open one: all run-#1 decisions are locked (§7), the agent has a stop condition (§6), and every rung has a named primary metric (§5). Let it rip.* ✍️🚀
