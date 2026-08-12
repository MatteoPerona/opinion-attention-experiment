"""Task 5 — Factored-logit diagnostic (architecture ablation; NOT a headline)."""

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
    joint_condition_number_report,
    ridge_fj_bundle,
    save_heatmaps,
    save_json,
    set_seed,
    train_fj,
)
from model.factored import FactoredFJAttentionHead  # noqa: E402
from sim import generate_dataset, make_world, train_val_split  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--M", type=int, default=2000)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--d", type=int, default=50)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="results/rung_3_factored_diag")
    args = p.parse_args()

    out = ensure_dir(ROOT / args.out)
    rng = set_seed(args.seed)
    world = make_world(N=args.N, rng=rng, kind="dense", degroot=False)
    ds = generate_dataset(world, M=args.M, T=args.T, sigma=args.sigma, rng=rng)
    train, val = train_val_split(ds, val_fraction=0.2, rng=rng)

    kappa = condition_number_report(train["pairs_X"])
    kappa_z = joint_condition_number_report(train["pairs_X"], train["pairs_x0"])
    ridge = ridge_fj_bundle(
        train["pairs_X"], train["pairs_Y"], train["pairs_x0"],
        world.W, world.Lambda, alpha=args.ridge_alpha,
    )

    model = FactoredFJAttentionHead(N=args.N, d=args.d)
    losses = train_fj(
        model, train["pairs_X"], train["pairs_Y"], train["pairs_x0"],
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )

    with torch.no_grad():
        A_full = model.attention_matrix().cpu().numpy()
        gates = model.gates().cpu().numpy()
    A_cur = A_full[:, : args.N]
    target_cur = np.diag(world.Lambda) @ world.W
    stub_true = 1.0 - world.Lambda

    attn_rel = relative_frobenius(A_cur, target_cur)
    # Also compare gates to 1-λ and (A_cur/(1-g)) to W
    stub_mae = float(np.mean(np.abs(gates - stub_true)))
    stub_corr = float(np.corrcoef(gates, stub_true)[0, 1])
    # Rescale current block by 1/(1-g) to compare to W
    scale = np.maximum(1.0 - gates, 1e-8)[:, None]
    A_as_W = A_cur / scale
    rel_vs_W = relative_frobenius(A_as_W, world.W)
    sp_vs_LW = spearman_corr_entries(A_cur, target_cur)

    pred_mse = eval_mse_fj(model, val["pairs_X"], val["pairs_Y"], val["pairs_x0"])

    # One-step joint-softmax reference from run #1
    joint_ref = {"attn_rel_f": 0.407289, "ridge_rel_f": 0.236520, "stub_corr": 0.999528}

    # Interpretation rule (fixed): if factored recovers near ridge, joint 2N-softmax was the bottleneck
    near_ridge = attn_rel < ridge["rel_frobenius_cur"] + 0.05
    if near_ridge:
        interpretation = (
            "Factored variant recovers ΛW near ridge → rung-3 gap was largely caused by "
            "the *joint* 2N-way softmax coupling current and anchor tokens."
        )
    else:
        interpretation = (
            "Factored variant still lags ridge → gap is not (only) the joint 2N-softmax; "
            "bilinear ID parameterization / optimization remains a bottleneck."
        )

    np.save(out / "true_W.npy", world.W)
    np.save(out / "true_Lambda.npy", world.Lambda)
    np.save(out / "learned_A.npy", A_full)
    np.save(out / "gates.npy", gates)
    np.save(out / "ridge_Op.npy", ridge["Op"])
    save_heatmaps(A_cur, target_cur, out / "heatmap_A_vs_W.png",
                  title_left="Factored A_cur", title_right="True ΛW")

    metrics = {
        "label": "ARCHITECTURE ABLATION — not a headline result",
        "attn_rel_frobenius_vs_LambdaW": attn_rel,
        "attn_rel_frobenius_vs_W_rescaled": rel_vs_W,
        "attn_spearman_vs_LambdaW": sp_vs_LW,
        "ridge_rel_frobenius_cur": ridge["rel_frobenius_cur"],
        "ridge_rel_frobenius_cur_simplex": ridge["rel_frobenius_cur_simplex"],
        "stubbornness_gate_mae": stub_mae,
        "stubbornness_gate_corr": stub_corr,
        "prediction_mse_heldout": pred_mse,
        "ridge_prediction_mse": ridge["pred_mse"],
        "state_cov_condition_number": kappa,
        "state_cov_condition_number_joint": kappa_z,
        "joint_softmax_ref": joint_ref,
        "near_ridge": near_ridge,
        "interpretation": interpretation,
        "final_train_loss": losses[-1],
    }
    save_json(metrics, out / "metrics.json")
    save_json({
        "task": 5,
        "diagnostic": True,
        "outside_run1_constraints": True,
        "N": args.N, "M": args.M, "T": args.T, "d": args.d,
        "sigma": args.sigma, "seed": args.seed, "epochs": args.epochs,
    }, out / "config.json")

    report = f"""# Task 5 Report — Factored-logit DIAGNOSTIC (not a headline)

> **Label:** architecture ablation outside the run-#1 constraint set.
> Do not cite as the main FJ recovery result.

## Purpose
Isolate whether the rung-3 A_cur↔ΛW gap is caused by the *joint* 2N-way softmax
(if factored recovers near ridge → yes) or by the bilinear ID parameterization itself
(if it doesn't → no).

## Setup
- σ={args.sigma}, M={args.M}, N={args.N}, d={args.d}, seed={args.seed}, epochs={args.epochs}
- Factored row: mass g_i on own anchor; (1−g_i)·softmax over N current tokens
- Raw-scalar values retained; no W_V / MLP

## Primary comparison

| Method | rel-F(A_cur vs ΛW) | stub corr | val MSE |
|--------|--------------------|-----------|---------|
| Joint 2N-softmax (run #1) | {joint_ref['attn_rel_f']:.4f} | {joint_ref['stub_corr']:.4f} | — |
| **Factored (this diag)** | **{attn_rel:.4f}** | **{stub_corr:.4f}** (gates) | {pred_mse:.4e} |
| Raw ridge | {ridge['rel_frobenius_cur']:.4f} | — | {ridge['pred_mse']:.4e} |
| Proj ridge | {ridge['rel_frobenius_cur_simplex']:.4f} | — | — |

- Spearman (factored A_cur vs ΛW) = {sp_vs_LW:.4f}
- Gate MAE vs (1−λ) = {stub_mae:.4f}
- Rescaled A_cur→W rel-F = {rel_vs_W:.4f}
- κ = {kappa:.4e}; κ_Z = {kappa_z:.4e}

## Heatmaps
`heatmap_A_vs_W.png`

## Interpretation
{interpretation}
"""
    (out / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
