"""Rung 3 — Friedkin–Johnsen + anchor tokens + noise (one change: stubbornness on)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import relative_frobenius, spearman_corr_entries  # noqa: E402
from experiments.common import (  # noqa: E402
    condition_number_report,
    ensure_dir,
    eval_mse_fj,
    ridge_fj_bundle,
    save_heatmaps,
    save_json,
    set_seed,
    train_fj,
)
from model import FJAnchorAttentionHead  # noqa: E402
from sim import generate_dataset, make_world, train_val_split  # noqa: E402


def run_one_seed(seed: int, args, out: Path) -> dict:
    rng = set_seed(seed)
    world = make_world(N=args.N, rng=rng, kind="dense", degroot=False)
    ds = generate_dataset(world, M=args.M, T=args.T, sigma=args.sigma, rng=rng)
    train, val = train_val_split(ds, val_fraction=0.2, rng=rng)

    cond = condition_number_report(train["pairs_X"])
    ridge = ridge_fj_bundle(
        train["pairs_X"],
        train["pairs_Y"],
        train["pairs_x0"],
        world.W,
        world.Lambda,
        alpha=args.ridge_alpha,
    )

    model = FJAnchorAttentionHead(N=args.N, d=args.d)
    losses = train_fj(
        model,
        train["pairs_X"],
        train["pairs_Y"],
        train["pairs_x0"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
    )
    with torch.no_grad():
        A_full = model.attention_matrix().cpu().numpy()
        A_cur = A_full[:, : args.N]
        A_anc = A_full[:, args.N :]
    target_cur = np.diag(world.Lambda) @ world.W
    target_anc = np.diag(1.0 - world.Lambda)

    # Primary dense metric on the current-opinion block vs ΛW
    attn_rel = relative_frobenius(A_cur, target_cur)
    attn_sp = spearman_corr_entries(A_cur, target_cur)
    # Also report recovery vs W after undoing known Λ scale on rows with λ>0
    A_as_W = A_cur / np.maximum(world.Lambda[:, None], 1e-8)
    # renormalize rows for comparison to W
    A_as_W = A_as_W / np.maximum(A_as_W.sum(axis=1, keepdims=True), 1e-8)
    rel_vs_W = relative_frobenius(A_as_W, world.W)

    stub_true = 1.0 - world.Lambda
    stub_pred = np.diag(A_anc)
    stub_mae = float(np.mean(np.abs(stub_pred - stub_true)))
    stub_corr = float(np.corrcoef(stub_pred, stub_true)[0, 1])

    pred_mse = eval_mse_fj(model, val["pairs_X"], val["pairs_Y"], val["pairs_x0"])

    seed_dir = ensure_dir(out / f"seed_{seed}")
    np.save(seed_dir / "true_W.npy", world.W)
    np.save(seed_dir / "true_Lambda.npy", world.Lambda)
    np.save(seed_dir / "learned_A.npy", A_full)
    np.save(seed_dir / "learned_A_current.npy", A_cur)
    np.save(seed_dir / "ridge_Op.npy", ridge["Op"])
    save_heatmaps(
        A_cur,
        target_cur,
        seed_dir / "heatmap_A_vs_W.png",
        title_left="Attention current block",
        title_right="True ΛW",
    )
    save_heatmaps(
        A_anc,
        target_anc,
        seed_dir / "heatmap_anchor.png",
        title_left="Attention anchor block",
        title_right="True (I-Λ)",
    )

    return {
        "seed": seed,
        "attn_rel_frobenius_vs_LambdaW": attn_rel,
        "attn_spearman_vs_LambdaW": attn_sp,
        "attn_rel_frobenius_vs_W_rescaled": rel_vs_W,
        "ridge_rel_frobenius_cur": ridge["rel_frobenius_cur"],
        "ridge_rel_frobenius_anc": ridge["rel_frobenius_anc"],
        "stubbornness_diag_mae": stub_mae,
        "stubbornness_corr": stub_corr,
        "prediction_mse_heldout": pred_mse,
        "ridge_prediction_mse": ridge["pred_mse"],
        "state_cov_condition_number": cond,
        "final_train_loss": losses[-1],
        "A_cur": A_cur,
        "A_full": A_full,
        "target_cur": target_cur,
        "W": world.W,
        "Lambda": world.Lambda,
        "ridge_Op": ridge["Op"],
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
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "lbfgs"])
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="results/rung_3")
    args = p.parse_args()

    out = ensure_dir(ROOT / args.out)
    seeds = [args.seed]
    if args.extra_seeds.strip():
        seeds = [int(s) for s in args.extra_seeds.split(",")]

    results = [run_one_seed(s, args, out) for s in seeds]
    primary = results[0]

    np.save(out / "true_W.npy", primary["W"])
    np.save(out / "true_Lambda.npy", primary["Lambda"])
    np.save(out / "learned_A.npy", primary["A_full"])
    np.save(out / "ridge_Op.npy", primary["ridge_Op"])
    save_heatmaps(
        primary["A_cur"],
        primary["target_cur"],
        out / "heatmap_A_vs_W.png",
        title_left="Attention current block",
        title_right="True ΛW",
    )

    config = {
        "rung": 3,
        "seeds": seeds,
        "N": args.N,
        "M": args.M,
        "T": args.T,
        "d": args.d,
        "sigma": args.sigma,
        "kind": "dense",
        "degroot": False,
        "epochs": args.epochs,
        "change_vs_rung_2": "FJ stubbornness + 2N anchor tokens",
    }
    save_json(config, out / "config.json")

    metrics = {
        "primary_metric": "rel_frobenius (attention current block vs ΛW)",
        "primary_value": primary["attn_rel_frobenius_vs_LambdaW"],
        "ridge_rel_frobenius_cur": primary["ridge_rel_frobenius_cur"],
        "stubbornness_diag_mae": primary["stubbornness_diag_mae"],
        "stubbornness_corr": primary["stubbornness_corr"],
        "prediction_mse_heldout": primary["prediction_mse_heldout"],
        "state_cov_condition_number": primary["state_cov_condition_number"],
        "per_seed": [
            {
                k: v
                for k, v in r.items()
                if k not in ("A_cur", "A_full", "target_cur", "W", "Lambda", "ridge_Op")
            }
            for r in results
        ],
    }
    save_json(metrics, out / "metrics.json")

    worse = primary["attn_rel_frobenius_vs_LambdaW"] > primary["ridge_rel_frobenius_cur"] * 2 + 0.05
    diagnosis = ""
    if worse:
        diagnosis = (
            "\n## Failure / gap diagnosis\n"
            "Attention current-block recovery worse than N×2N ridge operator. "
            "Possible causes: (1) optimization — 2N-way softmax harder; "
            "(2) architecture — Q/K from IDs must jointly represent ΛW and diagonal anchors; "
            "(3) capacity OK if d≥N. "
            + (
                "Multi-seed already run.\n"
                if len(seeds) > 1
                else "Consider `--extra-seeds 0,1,2`.\n"
            )
        )

    report = f"""# Rung 3 Report — FJ + anchor tokens + noise

## Setup
- Exactly one change vs rung 2: **stubbornness on** (Λ≠I) with **2N anchor-token** design
- σ={args.sigma} retained; dense W; N={args.N}, M={args.M}, T={args.T}, d={args.d}
- Seeds={seeds}

## Primary metric (§5 — dense: relative Frobenius on current block vs ΛW)
- **Attention** \(\\|A_{{cur}} - \\Lambda W\\|_F / \\|\\Lambda W\\|_F\) = **{primary["attn_rel_frobenius_vs_LambdaW"]:.6f}**
- Spearman = {primary["attn_spearman_vs_LambdaW"]:.6f}
- Rescaled-to-W rel-Frobenius = {primary["attn_rel_frobenius_vs_W_rescaled"]:.6f}

## Ridge-baseline recovery (same data, N×2N operator)
- Ridge current-block vs ΛW = **{primary["ridge_rel_frobenius_cur"]:.6f}**
- Ridge anchor-block vs (I−Λ) = {primary["ridge_rel_frobenius_anc"]:.6f}

## Stubbornness recovery (bonus)
- MAE of diag(A_anc) vs (1−λ) = **{primary["stubbornness_diag_mae"]:.6f}**
- Correlation = {primary["stubbornness_corr"]:.6f}

## Prediction MSE (held-out)
- Attention = **{primary["prediction_mse_heldout"]:.6e}**
- Ridge = {primary["ridge_prediction_mse"]:.6e}

## State-covariance condition number
- κ = **{primary["state_cov_condition_number"]:.6e}**

## Heatmaps
`heatmap_A_vs_W.png` (current block vs ΛW); per-seed also has anchor heatmap.
{diagnosis}
"""
    (out / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
