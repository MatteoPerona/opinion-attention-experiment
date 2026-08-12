"""Rung 2 — DeGroot + process noise (Run #2: 3 seeds + simplex-projected ridge)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import (  # noqa: E402
    condition_number_report,
    degroot_metrics,
    ensure_dir,
    eval_mse_degroot,
    ridge_degroot_bundle,
    save_heatmaps,
    save_json,
    set_seed,
    train_degroot,
)
from model import IdentityAttentionHead  # noqa: E402
from sim import generate_dataset, make_world, train_val_split  # noqa: E402

# Run #1 single-seed numbers (kept for continuity after overwrite)
RUN1_SEED0 = {
    "attn_rel_frobenius": 0.189623,
    "ridge_rel_frobenius_raw": 0.200684,
    "prediction_mse_heldout": 2.580179e-03,
    "ridge_prediction_mse_heldout": 2.590654e-03,
    "state_cov_condition_number": 6.135020e01,
    "note": "run #1 seed 0 only; raw ridge (no simplex projection)",
}


def run_one_seed(seed: int, args, out: Path) -> dict:
    rng = set_seed(seed)
    world = make_world(N=args.N, rng=rng, kind="dense", degroot=True)
    ds = generate_dataset(world, M=args.M, T=args.T, sigma=args.sigma, rng=rng)
    train, val = train_val_split(ds, val_fraction=0.2, rng=rng)

    cond = condition_number_report(train["pairs_X"])
    ridge = ridge_degroot_bundle(
        train["pairs_X"], train["pairs_Y"], world.W, alpha=args.ridge_alpha
    )
    ridge_val_mse = float(
        np.mean((val["pairs_X"] @ ridge["W_hat"].T - val["pairs_Y"]) ** 2)
    )
    ridge_simplex_val_mse = float(
        np.mean((val["pairs_X"] @ ridge["W_hat_simplex"].T - val["pairs_Y"]) ** 2)
    )

    model = IdentityAttentionHead(N=args.N, d=args.d)
    losses = train_degroot(
        model,
        train["pairs_X"],
        train["pairs_Y"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    with torch.no_grad():
        A = model.attention_matrix().cpu().numpy()
    attn_m = degroot_metrics(A, world.W)
    pred_mse = eval_mse_degroot(model, val["pairs_X"], val["pairs_Y"])

    seed_dir = ensure_dir(out / f"seed_{seed}")
    np.save(seed_dir / "true_W.npy", world.W)
    np.save(seed_dir / "true_Lambda.npy", world.Lambda)
    np.save(seed_dir / "learned_A.npy", A)
    np.save(seed_dir / "ridge_W.npy", ridge["W_hat"])
    np.save(seed_dir / "ridge_W_simplex.npy", ridge["W_hat_simplex"])
    save_heatmaps(A, world.W, seed_dir / "heatmap_A_vs_W.png")
    save_heatmaps(
        ridge["W_hat_simplex"],
        world.W,
        seed_dir / "heatmap_ridge_simplex_vs_W.png",
        title_left="Ridge simplex-proj",
        title_right="True W",
    )

    return {
        "seed": seed,
        "attn_rel_frobenius": attn_m["rel_frobenius"],
        "attn_spearman": attn_m["spearman"],
        "ridge_rel_frobenius_raw": ridge["rel_frobenius_raw"],
        "ridge_rel_frobenius_simplex": ridge["rel_frobenius_simplex"],
        "ridge_spearman_raw": ridge["spearman_raw"],
        "ridge_spearman_simplex": ridge["spearman_simplex"],
        "prediction_mse_heldout": pred_mse,
        "ridge_prediction_mse_heldout": ridge_val_mse,
        "ridge_simplex_prediction_mse_heldout": ridge_simplex_val_mse,
        "state_cov_condition_number": cond,
        "final_train_loss": losses[-1],
        "A": A,
        "W": world.W,
        "Lambda": world.Lambda,
        "ridge_W": ridge["W_hat"],
        "ridge_W_simplex": ridge["W_hat_simplex"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra-seeds", type=str, default="0,1,2")
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--M", type=int, default=200)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--d", type=int, default=50)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="results/rung_2")
    args = p.parse_args()

    out = ensure_dir(ROOT / args.out)
    seeds = [int(s) for s in args.extra_seeds.split(",")] if args.extra_seeds.strip() else [args.seed]

    results = [run_one_seed(s, args, out) for s in seeds]
    primary = results[0]

    np.save(out / "true_W.npy", primary["W"])
    np.save(out / "true_Lambda.npy", primary["Lambda"])
    np.save(out / "learned_A.npy", primary["A"])
    np.save(out / "ridge_W.npy", primary["ridge_W"])
    np.save(out / "ridge_W_simplex.npy", primary["ridge_W_simplex"])
    save_heatmaps(primary["A"], primary["W"], out / "heatmap_A_vs_W.png")

    attn_vals = [r["attn_rel_frobenius"] for r in results]
    raw_vals = [r["ridge_rel_frobenius_raw"] for r in results]
    proj_vals = [r["ridge_rel_frobenius_simplex"] for r in results]
    mean_attn = float(np.mean(attn_vals))
    mean_raw = float(np.mean(raw_vals))
    mean_proj = float(np.mean(proj_vals))

    # Decision rule (fixed before looking): confirmed iff mean attn <= mean projected ridge
    claim_confirmed = mean_attn <= mean_proj
    if claim_confirmed:
        verdict = (
            f"**CONFIRMED:** mean attention rel-F ({mean_attn:.6f}) ≤ mean "
            f"simplex-projected ridge rel-F ({mean_proj:.6f}). "
            "Attention matches or beats the fair (row-stochastic) ridge comparator under noise."
        )
    else:
        verdict = (
            f"**RECLASSIFIED:** mean projected-ridge rel-F ({mean_proj:.6f}) < mean "
            f"attention ({mean_attn:.6f}). Run #1's 'attention beats ridge' was an artifact of "
            "comparing softmax-constrained attention to *unconstrained* ridge. The fair "
            "sentence is: attention's advantage was the row-stochasticity constraint itself."
        )

    config = {
        "rung": 2,
        "run": 2,
        "seeds": seeds,
        "N": args.N,
        "M": args.M,
        "T": args.T,
        "d": args.d,
        "sigma": args.sigma,
        "kind": "dense",
        "degroot": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "ridge_alpha": args.ridge_alpha,
        "decision_rule": "confirmed iff mean_attn_rel_F <= mean_projected_ridge_rel_F",
        "run1_seed0_archived": RUN1_SEED0,
    }
    save_json(config, out / "config.json")

    slim = [
        {k: v for k, v in r.items() if k not in ("A", "W", "Lambda", "ridge_W", "ridge_W_simplex")}
        for r in results
    ]
    metrics = {
        "primary_metric": "rel_frobenius (attention A vs true W)",
        "primary_value": primary["attn_rel_frobenius"],
        "mean_attn_rel_frobenius": mean_attn,
        "mean_ridge_rel_frobenius_raw": mean_raw,
        "mean_ridge_rel_frobenius_simplex": mean_proj,
        "claim_confirmed_vs_projected_ridge": claim_confirmed,
        "prediction_mse_heldout": primary["prediction_mse_heldout"],
        "state_cov_condition_number": primary["state_cov_condition_number"],
        "run1_seed0_archived": RUN1_SEED0,
        "per_seed": slim,
    }
    save_json(metrics, out / "metrics.json")

    table_rows = "\n".join(
        f"| {r['seed']} | {r['attn_rel_frobenius']:.6f} | {r['ridge_rel_frobenius_raw']:.6f} | "
        f"{r['ridge_rel_frobenius_simplex']:.6f} | {r['prediction_mse_heldout']:.6e} | "
        f"{r['state_cov_condition_number']:.4e} |"
        for r in results
    )

    report = f"""# Rung 2 Report (Run #2) — DeGroot + process noise, 3 seeds + projected ridge

## Setup
- Identical to run #1 rung 2: σ={args.sigma}, N={args.N}, M={args.M}, T={args.T}, d={args.d}, dense DeGroot
- Seeds={seeds}, epochs={args.epochs}, lr={args.lr}
- Comparators: attention, **raw ridge**, **Euclidean simplex-projected ridge**

## Run #1 continuity (archived seed-0 numbers before overwrite)
- Attention rel-F = {RUN1_SEED0['attn_rel_frobenius']:.6f}
- Raw ridge rel-F = {RUN1_SEED0['ridge_rel_frobenius_raw']:.6f}
- (No projected-ridge comparator in run #1)

## Decision rule (fixed before results)
Confirm run-#1 claim "attention matches or beats ridge under noise" **only if**
mean attention rel-F ≤ mean *projected*-ridge rel-F across 3 seeds.
Otherwise reclassify: advantage was the row-stochasticity constraint itself.

## Primary metric (§5 — dense: relative Frobenius)

| seed | attention | raw ridge | simplex-proj ridge | attn val MSE | κ |
|------|-----------|-----------|--------------------|--------------|---|
{table_rows}
| **mean** | **{mean_attn:.6f}** | **{mean_raw:.6f}** | **{mean_proj:.6f}** | — | — |

## Prediction MSE (held-out, seed 0)
- Attention = **{primary["prediction_mse_heldout"]:.6e}**
- Raw ridge = {primary["ridge_prediction_mse_heldout"]:.6e}
- Simplex-proj ridge = {primary["ridge_simplex_prediction_mse_heldout"]:.6e}

## State-covariance condition number (seed 0)
- κ(cov(X)) = **{primary["state_cov_condition_number"]:.6e}**

## Heatmaps
`heatmap_A_vs_W.png`; per-seed also has `heatmap_ridge_simplex_vs_W.png`.

## Verdict
{verdict}
"""
    (out / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
