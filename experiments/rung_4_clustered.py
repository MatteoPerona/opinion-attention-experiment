"""Rung 4 — clustered/sparse graph structure (one change vs rung 3: graph kind)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import (  # noqa: E402
    relative_frobenius,
    spearman_corr_entries,
    topk_edge_precision,
)
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
    world = make_world(
        N=args.N,
        rng=rng,
        kind=args.kind,
        degroot=False,
        n_clusters=args.n_clusters,
    )
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
    # Primary for sparse/clustered: Spearman + top-k on current block vs ΛW
    # Also report vs W structure (edges of W)
    sp_LW = spearman_corr_entries(A_cur, target_cur)
    sp_W = spearman_corr_entries(A_cur, world.W)
    k_edges = int((world.W > 1e-8).sum())
    topk_W = topk_edge_precision(A_cur, world.W, k=k_edges)
    topk_LW = topk_edge_precision(A_cur, target_cur, k=k_edges)
    rel_f = relative_frobenius(A_cur, target_cur)

    stub_true = 1.0 - world.Lambda
    stub_pred = np.diag(A_anc)
    stub_mae = float(np.mean(np.abs(stub_pred - stub_true)))

    pred_mse = eval_mse_fj(model, val["pairs_X"], val["pairs_Y"], val["pairs_x0"])

    seed_dir = ensure_dir(out / f"seed_{seed}")
    np.save(seed_dir / "true_W.npy", world.W)
    np.save(seed_dir / "true_Lambda.npy", world.Lambda)
    np.save(seed_dir / "learned_A.npy", A_full)
    np.save(seed_dir / "learned_A_current.npy", A_cur)
    np.save(seed_dir / "ridge_Op.npy", ridge["Op"])
    save_heatmaps(
        A_cur,
        world.W,
        seed_dir / "heatmap_A_vs_W.png",
        title_left="Attention current block",
        title_right="True W (clustered)",
    )

    return {
        "seed": seed,
        "attn_spearman_vs_LambdaW": sp_LW,
        "attn_spearman_vs_W": sp_W,
        "attn_topk_precision_vs_W": topk_W,
        "attn_topk_precision_vs_LambdaW": topk_LW,
        "attn_rel_frobenius_vs_LambdaW": rel_f,
        "ridge_rel_frobenius_cur": ridge["rel_frobenius_cur"],
        "ridge_spearman_cur": ridge["spearman_cur"],
        "stubbornness_diag_mae": stub_mae,
        "prediction_mse_heldout": pred_mse,
        "ridge_prediction_mse": ridge["pred_mse"],
        "state_cov_condition_number": cond,
        "n_true_edges": k_edges,
        "final_train_loss": losses[-1],
        "A_cur": A_cur,
        "A_full": A_full,
        "W": world.W,
        "Lambda": world.Lambda,
        "ridge_Op": ridge["Op"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra-seeds", type=str, default="")
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--M", type=int, default=2000)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--d", type=int, default=50)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--kind", type=str, default="clustered", choices=["clustered", "sparse"])
    p.add_argument("--n-clusters", type=int, default=5)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "lbfgs"])
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="results/rung_4")
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
        primary["W"],
        out / "heatmap_A_vs_W.png",
        title_left="Attention current block",
        title_right=f"True W ({args.kind})",
    )

    config = {
        "rung": 4,
        "seeds": seeds,
        "N": args.N,
        "M": args.M,
        "T": args.T,
        "d": args.d,
        "sigma": args.sigma,
        "kind": args.kind,
        "n_clusters": args.n_clusters,
        "degroot": False,
        "epochs": args.epochs,
        "change_vs_rung_3": f"graph structure -> {args.kind}",
        "primary_metrics": "Spearman + top-k edge precision",
    }
    save_json(config, out / "config.json")

    metrics = {
        "primary_metric": "spearman + topk_precision (attention current vs W / ΛW)",
        "primary_spearman_vs_W": primary["attn_spearman_vs_W"],
        "primary_topk_vs_W": primary["attn_topk_precision_vs_W"],
        "attn_spearman_vs_LambdaW": primary["attn_spearman_vs_LambdaW"],
        "attn_rel_frobenius_vs_LambdaW": primary["attn_rel_frobenius_vs_LambdaW"],
        "ridge_rel_frobenius_cur": primary["ridge_rel_frobenius_cur"],
        "ridge_spearman_cur": primary["ridge_spearman_cur"],
        "prediction_mse_heldout": primary["prediction_mse_heldout"],
        "state_cov_condition_number": primary["state_cov_condition_number"],
        "per_seed": [
            {
                k: v
                for k, v in r.items()
                if k not in ("A_cur", "A_full", "W", "Lambda", "ridge_Op")
            }
            for r in results
        ],
    }
    save_json(metrics, out / "metrics.json")

    # Failure: attention Spearman much worse than ridge Spearman on current block
    worse = primary["attn_spearman_vs_LambdaW"] < primary["ridge_spearman_cur"] - 0.15
    diagnosis = ""
    if worse:
        diagnosis = (
            "\n## Failure / gap diagnosis\n"
            "Attention rank-correlation below ridge on ΛW. Clustered graphs induce "
            "within-cluster collinearity (§2) — individual edges mushy even for ridge. "
            "Softmax positivity floor also hurts Frobenius; primary is Spearman/top-k.\n"
        )

    report = f"""# Rung 4 Report — {args.kind} graph + FJ + noise

## Setup
- Exactly one change vs rung 3: **graph kind = {args.kind}** (n_clusters={args.n_clusters})
- FJ + σ={args.sigma}, N={args.N}, M={args.M}, T={args.T}, d={args.d}
- Seeds={seeds}
- True edges k={primary["n_true_edges"]}

## Primary metrics (§5 — sparse/clustered: Spearman + top-k)
- **Spearman (A_cur vs W)** = **{primary["attn_spearman_vs_W"]:.6f}**
- **Top-k precision (A_cur vs W)** = **{primary["attn_topk_precision_vs_W"]:.6f}**
- Spearman vs ΛW = {primary["attn_spearman_vs_LambdaW"]:.6f}
- Top-k vs ΛW = {primary["attn_topk_precision_vs_LambdaW"]:.6f}
- Rel-Frobenius vs ΛW (secondary; positivity floor) = {primary["attn_rel_frobenius_vs_LambdaW"]:.6f}

## Ridge-baseline recovery (same data)
- Ridge rel-Frobenius (current vs ΛW) = **{primary["ridge_rel_frobenius_cur"]:.6f}**
- Ridge Spearman (current vs ΛW) = **{primary["ridge_spearman_cur"]:.6f}**

## Stubbornness recovery
- MAE diag(A_anc) vs (1−λ) = {primary["stubbornness_diag_mae"]:.6f}

## Prediction MSE (held-out)
- Attention = **{primary["prediction_mse_heldout"]:.6e}**
- Ridge = {primary["ridge_prediction_mse"]:.6e}

## State-covariance condition number
- κ = **{primary["state_cov_condition_number"]:.6e}**

## Heatmaps
`heatmap_A_vs_W.png`
{diagnosis}
"""
    (out / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
