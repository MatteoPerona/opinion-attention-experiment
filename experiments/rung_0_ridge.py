"""Rung 0 — ridge-regression baseline (no neural net)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import (  # noqa: E402
    condition_number_report,
    ensure_dir,
    ridge_degroot_bundle,
    save_heatmaps,
    save_json,
    set_seed,
)
from sim import generate_dataset, make_world, train_val_split  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--M", type=int, default=200)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="results/rung_0")
    args = p.parse_args()

    out = ensure_dir(ROOT / args.out)
    rng = set_seed(args.seed)

    world = make_world(N=args.N, rng=rng, kind="dense", degroot=True)
    ds = generate_dataset(world, M=args.M, T=args.T, sigma=args.sigma, rng=rng)
    train, val = train_val_split(ds, val_fraction=0.2, rng=rng)

    cond = condition_number_report(train["pairs_X"])
    ridge = ridge_degroot_bundle(
        train["pairs_X"], train["pairs_Y"], world.W, alpha=args.ridge_alpha
    )
    # held-out prediction with ridge W
    val_mse = float(np.mean((val["pairs_X"] @ ridge["W_hat"].T - val["pairs_Y"]) ** 2))

    # Adaptive M note
    primary = ridge["rel_frobenius_raw"]
    adaptive_note = ""
    M_used = args.M
    if primary > 0.15:
        # raise M and retry once
        M_used = args.M * 4
        adaptive_note = (
            f"Initial ridge rel-Frobenius={primary:.4f} with M={args.M} was poor; "
            f"raised M to {M_used} per §2 adaptive rule and re-fit."
        )
        rng2 = set_seed(args.seed)
        world = make_world(N=args.N, rng=rng2, kind="dense", degroot=True)
        ds = generate_dataset(world, M=M_used, T=args.T, sigma=args.sigma, rng=rng2)
        train, val = train_val_split(ds, val_fraction=0.2, rng=rng2)
        cond = condition_number_report(train["pairs_X"])
        ridge = ridge_degroot_bundle(
            train["pairs_X"], train["pairs_Y"], world.W, alpha=args.ridge_alpha
        )
        val_mse = float(np.mean((val["pairs_X"] @ ridge["W_hat"].T - val["pairs_Y"]) ** 2))
        primary = ridge["rel_frobenius_raw"]

    np.save(out / "true_W.npy", world.W)
    np.save(out / "true_Lambda.npy", world.Lambda)
    np.save(out / "ridge_W.npy", ridge["W_hat"])
    np.save(out / "ridge_W_stoch.npy", ridge["W_hat_stoch"])
    np.save(out / "learned_A.npy", ridge["W_hat"])  # for heatmap protocol uniformity

    save_heatmaps(
        ridge["W_hat"],
        world.W,
        out / "heatmap_A_vs_W.png",
        title_left="Ridge W_hat",
        title_right="True W",
    )

    config = {
        "rung": 0,
        "seed": args.seed,
        "N": args.N,
        "M": M_used,
        "T": args.T,
        "sigma": args.sigma,
        "kind": "dense",
        "degroot": True,
        "ridge_alpha": args.ridge_alpha,
        "adaptive_note": adaptive_note,
    }
    save_json(config, out / "config.json")

    metrics = {
        "primary_metric": "rel_frobenius (ridge vs true W)",
        "primary_value": primary,
        "ridge_rel_frobenius_raw": ridge["rel_frobenius_raw"],
        "ridge_rel_frobenius_stoch": ridge["rel_frobenius_stoch"],
        "ridge_spearman_raw": ridge["spearman_raw"],
        "ridge_pred_mse_train": ridge["pred_mse"],
        "prediction_mse_heldout": val_mse,
        "state_cov_condition_number": cond,
        "n_train_pairs": int(train["pairs_X"].shape[0]),
        "n_val_pairs": int(val["pairs_X"].shape[0]),
    }
    save_json(metrics, out / "metrics.json")

    report = f"""# Rung 0 Report — Ridge-regression baseline

## Setup
- N={args.N}, M={M_used}, T={args.T}, sigma={args.sigma}, dense DeGroot (Λ=I)
- Seed={args.seed}, ridge α={args.ridge_alpha}
- Training pairs (held-in trajectories): {train["pairs_X"].shape[0]}
{("- " + adaptive_note) if adaptive_note else ""}

## Primary metric (§5 — dense: relative Frobenius)
- **Ridge recovery** \(\\|\\hat{{W}} - W\\|_F / \\|W\\|_F\) = **{primary:.6f}**
- Row-softmax-projected ridge rel-Frobenius = {ridge["rel_frobenius_stoch"]:.6f}
- Spearman (raw ridge vs W) = {ridge["spearman_raw"]:.6f}

## Ridge-baseline recovery (this is the baseline)
Same as primary: {primary:.6f}

## Prediction MSE (held-out trajectories)
- Held-out MSE with \(\\hat{{W}}\): **{val_mse:.6e}**
- Train MSE: {ridge["pred_mse"]:.6e}

## State-covariance condition number
- κ(cov(X)) = **{cond:.6e}**

## Artifacts
- `true_W.npy`, `true_Lambda.npy`, `ridge_W.npy`, `heatmap_A_vs_W.png`, `config.json`, `metrics.json`

## Verdict
{"PASS: data supports recovery (ridge rel-Frobenius < 0.1)." if primary < 0.1 else "WARN: ridge recovery mediocre — consider raising M further before attention." if primary < 0.3 else "FAIL: data under-exciting; do not trust attention results until M/excitation fixed."}

Gate for rung 1: ridge must show the data identifies W. Attention cannot beat missing information.
"""
    (out / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
