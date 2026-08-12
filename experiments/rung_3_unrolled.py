"""Task 4 — k-step unrolled training on FJ rung-3 configuration (§4 graduation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    train_fj_unrolled,
)
from model import FJAnchorAttentionHead  # noqa: E402
from sim import generate_dataset, make_world, train_val_split  # noqa: E402

# Success criterion fixed BEFORE running
SUCCESS_CRITERION = {
    "one_step_baseline_rel_f": 0.40,
    "seed_spread": 0.05,
    "must_improve_by_more_than": 0.05,
    "interesting_threshold_ridge": 0.23,
    "stubbornness_corr_floor": 0.99,
    "rule": (
        "Success if mean unrolled attn rel-F(A_cur vs ΛW) improves on one-step ≈0.40 "
        "by a margin larger than seed spread ≈0.05 (i.e. mean < 0.35); interesting if "
        "approaching ridge ≈0.23. Stubbornness corr must stay >0.99."
    ),
}

# Run #1 one-step reference (seed means from rung_3 REPORT)
ONE_STEP_REF = {
    "attn_rel_f_seed0": 0.407289,
    "attn_rel_f_seeds": [0.407, 0.370, 0.421],
    "ridge_rel_f_seed0": 0.236520,
    "stub_corr_seed0": 0.999528,
}


def curriculum_train(model, train_traj, train_x0, args) -> dict:
    """k=1 (>=300 epochs) → k=2 → k=4 with grad clip; drop lr if unstable."""
    schedule = [
        {"k": 1, "epochs": args.epochs_k1, "lr": args.lr},
        {"k": 2, "epochs": args.epochs_k2, "lr": args.lr},
        {"k": 4, "epochs": args.epochs_k4, "lr": args.lr * 0.5},
    ]
    all_losses = {}
    unstable = False
    unstable_note = ""

    for stage in schedule:
        k = stage["k"]
        lr = stage["lr"]
        print(f"  curriculum k={k} epochs={stage['epochs']} lr={lr}", flush=True)
        losses = train_fj_unrolled(
            model,
            train_traj,
            train_x0,
            k=k,
            epochs=stage["epochs"],
            batch_size=args.batch_size,
            lr=lr,
            grad_clip=args.grad_clip,
        )
        # detect instability: NaN or explosion
        if not np.isfinite(losses[-1]) or losses[-1] > max(losses[0] * 50, 10.0):
            # retry once with lower lr
            lr2 = lr * 0.2
            print(f"  instability at k={k} (loss={losses[-1]}); retry lr={lr2}", flush=True)
            losses2 = train_fj_unrolled(
                model,
                train_traj,
                train_x0,
                k=k,
                epochs=stage["epochs"],
                batch_size=args.batch_size,
                lr=lr2,
                grad_clip=args.grad_clip,
            )
            if not np.isfinite(losses2[-1]) or losses2[-1] > max(losses2[0] * 50, 10.0):
                unstable = True
                unstable_note = f"Unstable at k={k} even after lr drop to {lr2}; final loss={losses2[-1]}"
                all_losses[f"k{k}"] = losses2
                break
            losses = losses2
        all_losses[f"k{k}"] = losses

    return {"losses": all_losses, "unstable": unstable, "unstable_note": unstable_note}


def run_one_seed(seed: int, args, out: Path) -> dict:
    rng = set_seed(seed)
    world = make_world(N=args.N, rng=rng, kind="dense", degroot=False)
    ds = generate_dataset(world, M=args.M, T=args.T, sigma=args.sigma, rng=rng)
    train, val = train_val_split(ds, val_fraction=0.2, rng=rng)

    kappa = condition_number_report(train["pairs_X"])
    kappa_z = joint_condition_number_report(train["pairs_X"], train["pairs_x0"])
    ridge = ridge_fj_bundle(
        train["pairs_X"],
        train["pairs_Y"],
        train["pairs_x0"],
        world.W,
        world.Lambda,
        alpha=args.ridge_alpha,
    )

    model = FJAnchorAttentionHead(N=args.N, d=args.d)
    train_info = curriculum_train(model, train["trajectories"], train["x0s"], args)

    with torch.no_grad():
        A_full = model.attention_matrix().cpu().numpy()
    A_cur = A_full[:, : args.N]
    A_anc = A_full[:, args.N :]
    target_cur = np.diag(world.Lambda) @ world.W
    stub_true = 1.0 - world.Lambda
    stub_pred = np.diag(A_anc)

    attn_rel = relative_frobenius(A_cur, target_cur)
    attn_sp = spearman_corr_entries(A_cur, target_cur)
    stub_mae = float(np.mean(np.abs(stub_pred - stub_true)))
    stub_corr = float(np.corrcoef(stub_pred, stub_true)[0, 1])
    pred_mse = eval_mse_fj(model, val["pairs_X"], val["pairs_Y"], val["pairs_x0"])

    seed_dir = ensure_dir(out / f"seed_{seed}")
    np.save(seed_dir / "true_W.npy", world.W)
    np.save(seed_dir / "true_Lambda.npy", world.Lambda)
    np.save(seed_dir / "learned_A.npy", A_full)
    np.save(seed_dir / "ridge_Op.npy", ridge["Op"])
    save_heatmaps(
        A_cur,
        target_cur,
        seed_dir / "heatmap_A_vs_W.png",
        title_left="Unrolled A_cur",
        title_right="True ΛW",
    )
    # loss curves
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, losses in train_info["losses"].items():
        ax.plot(losses, label=name)
    ax.set_xlabel("epoch (within stage)")
    ax.set_ylabel("unrolled MSE")
    ax.set_title(f"seed {seed} curriculum losses")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(seed_dir / "loss_curves.png", dpi=120)
    plt.close(fig)
    save_json(
        {
            "losses_tail": {k: v[-20:] for k, v in train_info["losses"].items()},
            "unstable": train_info["unstable"],
            "unstable_note": train_info["unstable_note"],
        },
        seed_dir / "train.json",
    )

    return {
        "seed": seed,
        "attn_rel_frobenius_vs_LambdaW": attn_rel,
        "attn_spearman_vs_LambdaW": attn_sp,
        "ridge_rel_frobenius_cur": ridge["rel_frobenius_cur"],
        "ridge_rel_frobenius_cur_simplex": ridge["rel_frobenius_cur_simplex"],
        "stubbornness_diag_mae": stub_mae,
        "stubbornness_corr": stub_corr,
        "prediction_mse_heldout": pred_mse,
        "ridge_prediction_mse": ridge["pred_mse"],
        "state_cov_condition_number": kappa,
        "state_cov_condition_number_joint": kappa_z,
        "unstable": train_info["unstable"],
        "unstable_note": train_info["unstable_note"],
        "A_cur": A_cur,
        "target_cur": target_cur,
        "W": world.W,
        "Lambda": world.Lambda,
        "A_full": A_full,
        "ridge_Op": ridge["Op"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extra-seeds", type=str, default="0,1,2")
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--M", type=int, default=2000)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--d", type=int, default=50)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--epochs-k1", type=int, default=300)
    p.add_argument("--epochs-k2", type=int, default=200)
    p.add_argument("--epochs-k4", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="results/rung_3_unrolled")
    args = p.parse_args()

    out = ensure_dir(ROOT / args.out)
    seeds = [int(s) for s in args.extra_seeds.split(",")]

    # Pre-register success criterion
    save_json(
        {"success_criterion": SUCCESS_CRITERION, "one_step_ref": ONE_STEP_REF},
        out / "success_criterion.json",
    )

    results = []
    for seed in seeds:
        print(f"=== Unrolled seed={seed} ===", flush=True)
        results.append(run_one_seed(seed, args, out))

    primary = results[0]
    np.save(out / "true_W.npy", primary["W"])
    np.save(out / "true_Lambda.npy", primary["Lambda"])
    np.save(out / "learned_A.npy", primary["A_full"])
    np.save(out / "ridge_Op.npy", primary["ridge_Op"])
    save_heatmaps(
        primary["A_cur"],
        primary["target_cur"],
        out / "heatmap_A_vs_W.png",
        title_left="Unrolled A_cur",
        title_right="True ΛW",
    )

    attn_vals = [r["attn_rel_frobenius_vs_LambdaW"] for r in results]
    mean_attn = float(np.mean(attn_vals))
    stub_corrs = [r["stubbornness_corr"] for r in results]
    improved = mean_attn < (SUCCESS_CRITERION["one_step_baseline_rel_f"] - SUCCESS_CRITERION["must_improve_by_more_than"])
    stub_ok = all(c > SUCCESS_CRITERION["stubbornness_corr_floor"] for c in stub_corrs if np.isfinite(c))
    interesting = mean_attn <= SUCCESS_CRITERION["interesting_threshold_ridge"] + 0.05
    any_unstable = any(r["unstable"] for r in results)

    if any_unstable:
        verdict = "UNSTABLE: training collapsed on at least one seed after curriculum/clip/lr drops. See loss curves."
    elif improved and stub_ok:
        verdict = (
            f"SUCCESS: mean unrolled rel-F={mean_attn:.4f} < 0.35 "
            f"(improved on one-step ~0.40 by >seed-spread); stub corr ok."
            + (" Approaches ridge threshold." if interesting else " Still above ridge≈0.23.")
        )
    elif stub_ok:
        verdict = (
            f"NO SUCCESS on criterion: mean unrolled rel-F={mean_attn:.4f} did not beat "
            f"one-step 0.40 by >0.05. Stubbornness preserved (corr>{SUCCESS_CRITERION['stubbornness_corr_floor']})."
        )
    else:
        verdict = f"FAILURE: stubbornness regressed (corrs={stub_corrs}). mean attn rel-F={mean_attn:.4f}."

    slim = [
        {k: v for k, v in r.items() if k not in ("A_cur", "target_cur", "W", "Lambda", "A_full", "ridge_Op")}
        for r in results
    ]
    metrics = {
        "primary_metric": "rel_frobenius (unrolled attention A_cur vs ΛW)",
        "mean_attn_rel_frobenius": mean_attn,
        "per_seed_attn_rel_frobenius": attn_vals,
        "success_criterion": SUCCESS_CRITERION,
        "one_step_ref": ONE_STEP_REF,
        "success": bool(improved and stub_ok and not any_unstable),
        "interesting_near_ridge": bool(interesting),
        "stubbornness_ok": bool(stub_ok),
        "verdict": verdict,
        "per_seed": slim,
    }
    save_json(metrics, out / "metrics.json")
    save_json(
        {
            "task": 4,
            "sigma": args.sigma,
            "M": args.M,
            "curriculum": {"k1": args.epochs_k1, "k2": args.epochs_k2, "k4": args.epochs_k4},
            "lr": args.lr,
            "grad_clip": args.grad_clip,
            "seeds": seeds,
            "success_criterion": SUCCESS_CRITERION,
        },
        out / "config.json",
    )

    table = "\n".join(
        f"| {r['seed']} | {r['attn_rel_frobenius_vs_LambdaW']:.4f} | {r['ridge_rel_frobenius_cur']:.4f} | "
        f"{r['ridge_rel_frobenius_cur_simplex']:.4f} | {r['stubbornness_corr']:.4f} | "
        f"{r['prediction_mse_heldout']:.4e} | {r['state_cov_condition_number']:.3e} | "
        f"{r['state_cov_condition_number_joint']:.3e} | {r['unstable']} |"
        for r in results
    )

    report = f"""# Task 4 Report — Unrolled FJ training (§4 graduation path)

## Setup
- Rung-3 config: dense FJ + 2N anchors, σ={args.sigma}, M={args.M}, N={args.N}, T={args.T}, d={args.d}
- Curriculum: k=1 ({args.epochs_k1} ep) → k=2 ({args.epochs_k2}) → k=4 ({args.epochs_k4}); grad clip={args.grad_clip}
- Anchors fixed at true x(0) throughout rollout; raw-scalar values; no new modules
- Seeds={seeds}

## Success criterion (fixed before running)
{SUCCESS_CRITERION['rule']}

## One-step vs unrolled (primary: rel-F A_cur vs ΛW)

| | Attention rel-F | Ridge rel-F | Stub corr |
|--|-----------------|-------------|-----------|
| Run #1 one-step (seed 0) | {ONE_STEP_REF['attn_rel_f_seed0']:.4f} | {ONE_STEP_REF['ridge_rel_f_seed0']:.4f} | {ONE_STEP_REF['stub_corr_seed0']:.4f} |
| Run #1 one-step (3 seeds) | {ONE_STEP_REF['attn_rel_f_seeds']} | — | — |
| **Unrolled mean** | **{mean_attn:.4f}** | — | — |

| seed | unrolled attn | raw ridge | proj ridge | stub corr | attn MSE | κ | κ_Z | unstable |
|------|---------------|-----------|------------|-----------|----------|---|-----|----------|
{table}

## Prediction MSE / condition numbers
See table; artifacts in `metrics.json`.

## Heatmaps / loss curves
`heatmap_A_vs_W.png`; per-seed `loss_curves.png`.

## Verdict
{verdict}
"""
    (out / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
