"""Rung 2 — DeGroot + process noise (exactly one change vs rung 1)."""

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
    save_heatmaps(A, world.W, seed_dir / "heatmap_A_vs_W.png")

    return {
        "seed": seed,
        "attn_rel_frobenius": attn_m["rel_frobenius"],
        "attn_spearman": attn_m["spearman"],
        "ridge_rel_frobenius": ridge["rel_frobenius_raw"],
        "ridge_spearman": ridge["spearman_raw"],
        "prediction_mse_heldout": pred_mse,
        "ridge_prediction_mse_heldout": ridge_val_mse,
        "state_cov_condition_number": cond,
        "final_train_loss": losses[-1],
        "A": A,
        "W": world.W,
        "Lambda": world.Lambda,
        "ridge_W": ridge["W_hat"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra-seeds", type=str, default="")
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--M", type=int, default=200)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--d", type=int, default=50)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="results/rung_2")
    args = p.parse_args()

    out = ensure_dir(ROOT / args.out)
    seeds = [args.seed]
    if args.extra_seeds.strip():
        seeds = [int(s) for s in args.extra_seeds.split(",")]

    results = [run_one_seed(s, args, out) for s in seeds]
    primary = results[0]

    np.save(out / "true_W.npy", primary["W"])
    np.save(out / "true_Lambda.npy", primary["Lambda"])
    np.save(out / "learned_A.npy", primary["A"])
    np.save(out / "ridge_W.npy", primary["ridge_W"])
    save_heatmaps(primary["A"], primary["W"], out / "heatmap_A_vs_W.png")

    config = {
        "rung": 2,
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
        "change_vs_rung_1": f"process noise sigma={args.sigma}",
    }
    save_json(config, out / "config.json")

    metrics = {
        "primary_metric": "rel_frobenius (attention A vs true W)",
        "primary_value": primary["attn_rel_frobenius"],
        "attn_spearman": primary["attn_spearman"],
        "ridge_rel_frobenius": primary["ridge_rel_frobenius"],
        "prediction_mse_heldout": primary["prediction_mse_heldout"],
        "state_cov_condition_number": primary["state_cov_condition_number"],
        "per_seed": [
            {k: v for k, v in r.items() if k not in ("A", "W", "Lambda", "ridge_W")}
            for r in results
        ],
    }
    save_json(metrics, out / "metrics.json")

    worse = primary["attn_rel_frobenius"] > primary["ridge_rel_frobenius"] * 2 + 0.05
    diagnosis = ""
    if worse:
        if len(seeds) == 1:
            diagnosis = (
                "\n## Failure protocol\n"
                "Attention substantially worse than ridge. Re-running diagnosis recommended "
                "with `--extra-seeds 0,1,2`.\n"
            )
        else:
            attn_vals = [r["attn_rel_frobenius"] for r in results]
            ridge_vals = [r["ridge_rel_frobenius"] for r in results]
            diagnosis = (
                "\n## Multi-seed diagnosis\n"
                f"- Attention: {attn_vals}\n- Ridge: {ridge_vals}\n"
                "- Likely optimization/architecture gap under noisy targets "
                "(attention constrained to softmax row-stochastic; ridge unconstrained).\n"
            )

    report = f"""# Rung 2 Report — DeGroot + process noise

## Setup
- Exactly one change vs rung 1: **σ = {args.sigma}** process noise
- N={args.N}, M={args.M}, T={args.T}, d={args.d}, dense DeGroot
- Seeds={seeds}, epochs={args.epochs}

## Primary metric (§5 — dense: relative Frobenius)
- **Attention recovery** = **{primary["attn_rel_frobenius"]:.6f}**
- Spearman = {primary["attn_spearman"]:.6f}

## Ridge-baseline recovery (same data)
- Ridge rel-Frobenius = **{primary["ridge_rel_frobenius"]:.6f}**

## Prediction MSE (held-out)
- Attention = **{primary["prediction_mse_heldout"]:.6e}**
- Ridge = {primary["ridge_prediction_mse_heldout"]:.6e}

## State-covariance condition number
- κ = **{primary["state_cov_condition_number"]:.6e}**

## Heatmaps
`heatmap_A_vs_W.png`
{diagnosis}
## Note
Noise may improve excitation (condition number) while raising irreducible prediction MSE.
"""
    (out / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
