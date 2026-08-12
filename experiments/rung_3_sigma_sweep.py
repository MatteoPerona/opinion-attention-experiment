"""Task 3 — σ-sweep on FJ rung-3 configuration (noise-robustness, §5)."""

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
    train_fj,
)
from model import FJAnchorAttentionHead  # noqa: E402
from sim import generate_dataset, make_world, train_val_split  # noqa: E402

# Hypotheses fixed BEFORE seeing results (also written into REPORT.md)
HYPOTHESES = {
    "i": "κ_Z falls as σ rises (noise decorrelates x(t) from x(0))",
    "ii": "ridge ΛW recovery is non-monotone in σ with an interior optimum (§1b)",
    "iii": "attention−ridge gap narrows as σ rises (flat current-vs-anchor dirs re-excited)",
}


def run_one(seed: int, sigma: float, args, out_dir: Path) -> dict:
    rng = set_seed(seed)
    world = make_world(N=args.N, rng=rng, kind="dense", degroot=False)
    ds = generate_dataset(world, M=args.M, T=args.T, sigma=sigma, rng=rng)
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
    losses = train_fj(
        model,
        train["pairs_X"],
        train["pairs_Y"],
        train["pairs_x0"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer="adam",
    )
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

    seed_dir = ensure_dir(out_dir / f"seed_{seed}")
    np.save(seed_dir / "true_W.npy", world.W)
    np.save(seed_dir / "true_Lambda.npy", world.Lambda)
    np.save(seed_dir / "learned_A.npy", A_full)
    np.save(seed_dir / "ridge_Op.npy", ridge["Op"])
    np.save(seed_dir / "ridge_Op_simplex.npy", ridge["Op_simplex"])
    save_heatmaps(
        A_cur,
        target_cur,
        seed_dir / "heatmap_A_vs_W.png",
        title_left="Attention A_cur",
        title_right="True ΛW",
    )
    save_json({"train_losses_tail": losses[-20:], "final_train_loss": losses[-1]}, seed_dir / "train.json")

    return {
        "seed": seed,
        "sigma": sigma,
        "attn_rel_frobenius_vs_LambdaW": attn_rel,
        "attn_spearman_vs_LambdaW": attn_sp,
        "ridge_rel_frobenius_cur": ridge["rel_frobenius_cur"],
        "ridge_rel_frobenius_cur_simplex": ridge["rel_frobenius_cur_simplex"],
        "ridge_spearman_cur": ridge["spearman_cur"],
        "stubbornness_diag_mae": stub_mae,
        "stubbornness_corr": stub_corr,
        "prediction_mse_heldout": pred_mse,
        "ridge_prediction_mse": ridge["pred_mse"],
        "state_cov_condition_number": kappa,
        "state_cov_condition_number_joint": kappa_z,
        "attn_minus_ridge_gap": attn_rel - ridge["rel_frobenius_cur"],
        "attn_minus_proj_ridge_gap": attn_rel - ridge["rel_frobenius_cur_simplex"],
        "final_train_loss": losses[-1],
    }


def make_plots(summary: list[dict], out: Path) -> None:
    # Prefer seed-0 curve for the line plot; multi-seed points shown as markers at 0.05, 0.2
    by_sigma = {}
    for r in summary:
        by_sigma.setdefault(r["sigma"], []).append(r)

    sigmas = sorted(by_sigma.keys())
    seed0 = [next(x for x in by_sigma[s] if x["seed"] == 0) for s in sigmas]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(sigmas, [r["attn_rel_frobenius_vs_LambdaW"] for r in seed0], "o-", label="attention", color="C0")
    ax.plot(sigmas, [r["ridge_rel_frobenius_cur"] for r in seed0], "s-", label="raw ridge", color="C1")
    ax.plot(
        sigmas,
        [r["ridge_rel_frobenius_cur_simplex"] for r in seed0],
        "^-",
        label="proj ridge",
        color="C2",
    )
    # multi-seed means at 0.05 and 0.2
    for s in (0.05, 0.2):
        if s in by_sigma and len(by_sigma[s]) > 1:
            for key, color, marker in (
                ("attn_rel_frobenius_vs_LambdaW", "C0", "o"),
                ("ridge_rel_frobenius_cur", "C1", "s"),
                ("ridge_rel_frobenius_cur_simplex", "C2", "^"),
            ):
                vals = [x[key] for x in by_sigma[s]]
                ax.errorbar(
                    s,
                    float(np.mean(vals)),
                    yerr=float(np.std(vals)),
                    fmt=marker,
                    color=color,
                    alpha=0.5,
                    capsize=3,
                )
    ax.set_xscale("log")
    ax.set_xlabel("σ (log)")
    ax.set_ylabel("rel-Frobenius vs ΛW")
    ax.set_title("Recovery vs process noise")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(sigmas, [r["state_cov_condition_number_joint"] for r in seed0], "o-", color="C3", label="κ_Z (joint)")
    ax.plot(sigmas, [r["state_cov_condition_number"] for r in seed0], "s--", color="C4", label="κ (marginal)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("σ (log)")
    ax.set_ylabel("condition number (log)")
    ax.set_title("Conditioning vs process noise")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "recovery_and_kappa_vs_sigma.png", dpi=150)
    plt.close(fig)


def verdict_hypotheses(seed0_rows: list[dict]) -> dict:
    sigmas = [r["sigma"] for r in seed0_rows]
    kz = [r["state_cov_condition_number_joint"] for r in seed0_rows]
    ridge = [r["ridge_rel_frobenius_cur"] for r in seed0_rows]
    gaps = [r["attn_minus_ridge_gap"] for r in seed0_rows]

    # (i) κ_Z falls as σ rises: check monotone-ish decrease (allow tiny noise)
    i_ok = all(kz[j] >= kz[j + 1] * 0.95 for j in range(len(kz) - 1)) or (
        kz[0] > kz[-1] and kz[0] / kz[-1] > 2
    )
    # stricter: first vs last and Spearman of -kz vs sigma
    from scipy.stats import spearmanr

    i_corr, _ = spearmanr(sigmas, kz)
    i_pass = i_corr < -0.5  # κ_Z decreases with σ

    # (ii) ridge non-monotone with interior optimum
    best_idx = int(np.argmin(ridge))
    ii_pass = 0 < best_idx < len(ridge) - 1

    # (iii) attention−ridge gap narrows as σ rises
    iii_corr, _ = spearmanr(sigmas, gaps)
    iii_pass = iii_corr < -0.3  # gap shrinks with σ

    return {
        "i": {"pass": bool(i_pass), "spearman_sigma_vs_kappa_z": float(i_corr), "kappa_z_by_sigma": list(zip(sigmas, kz))},
        "ii": {
            "pass": bool(ii_pass),
            "best_sigma": sigmas[best_idx],
            "best_ridge_rel_f": ridge[best_idx],
            "ridge_by_sigma": list(zip(sigmas, ridge)),
        },
        "iii": {
            "pass": bool(iii_pass),
            "spearman_sigma_vs_gap": float(iii_corr),
            "gap_by_sigma": list(zip(sigmas, gaps)),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--M", type=int, default=2000)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--d", type=int, default=50)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="results/rung_3_sigma_sweep")
    p.add_argument(
        "--sigmas",
        type=str,
        default="0.0,0.02,0.05,0.1,0.2,0.35,0.5",
    )
    args = p.parse_args()

    out = ensure_dir(ROOT / args.out)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    # seed 0 everywhere; {0,1,2} at 0.05 and 0.2
    jobs = []
    for sigma in sigmas:
        seeds = [0, 1, 2] if abs(sigma - 0.05) < 1e-12 or abs(sigma - 0.2) < 1e-12 else [0]
        for seed in seeds:
            jobs.append((seed, sigma))

    # Write hypotheses BEFORE running (checkpoint integrity)
    pre_report = ensure_dir(out)
    save_json({"hypotheses_pre_registered": HYPOTHESES, "jobs": [{"seed": s, "sigma": g} for s, g in jobs]}, out / "hypotheses.json")

    summary = []
    for seed, sigma in jobs:
        tag = f"sigma_{sigma:g}"
        sub = ensure_dir(out / tag)
        print(f"=== Running seed={seed} sigma={sigma} ===", flush=True)
        r = run_one(seed, sigma, args, sub)
        summary.append(r)
        save_json(r, sub / f"seed_{seed}_metrics.json")

    save_json({"runs": summary}, out / "metrics.json")
    make_plots(summary, out)

    seed0_rows = sorted([r for r in summary if r["seed"] == 0], key=lambda x: x["sigma"])
    verdicts = verdict_hypotheses(seed0_rows)
    save_json(verdicts, out / "hypothesis_verdicts.json")

    config = {
        "task": 3,
        "setting": "rung-3 FJ + 2N anchors, vary only sigma",
        "N": args.N,
        "M": args.M,
        "T": args.T,
        "d": args.d,
        "epochs": args.epochs,
        "lr": args.lr,
        "sigmas": sigmas,
        "multi_seed_sigmas": [0.05, 0.2],
        "hypotheses_pre_registered": HYPOTHESES,
    }
    save_json(config, out / "config.json")

    # Build table
    rows = []
    for r in seed0_rows:
        rows.append(
            f"| {r['sigma']:g} | {r['attn_rel_frobenius_vs_LambdaW']:.4f} | "
            f"{r['ridge_rel_frobenius_cur']:.4f} | {r['ridge_rel_frobenius_cur_simplex']:.4f} | "
            f"{r['attn_spearman_vs_LambdaW']:.4f} | {r['stubbornness_corr']:.4f} | "
            f"{r['prediction_mse_heldout']:.4e} | {r['state_cov_condition_number']:.3e} | "
            f"{r['state_cov_condition_number_joint']:.3e} | {r['attn_minus_ridge_gap']:.4f} |"
        )

    def vline(key: str) -> str:
        v = verdicts[key]
        status = "SUPPORTED" if v["pass"] else "NOT SUPPORTED"
        return f"**({key}) {HYPOTHESES[key]}** — **{status}**. Details: `{v}`"

    # multi-seed notes
    ms_notes = []
    for s in (0.05, 0.2):
        group = [r for r in summary if abs(r["sigma"] - s) < 1e-12]
        if len(group) > 1:
            ms_notes.append(
                f"- σ={s}: attn rel-F={[round(g['attn_rel_frobenius_vs_LambdaW'],4) for g in group]}; "
                f"ridge={[round(g['ridge_rel_frobenius_cur'],4) for g in group]}; "
                f"κ_Z={[round(g['state_cov_condition_number_joint'],1) for g in group]}"
            )

    report = f"""# Task 3 Report — FJ σ-sweep (rung-3 configuration)

## Setup
- Dense FJ + 2N anchor tokens; N={args.N}, M={args.M}, T={args.T}, d={args.d}
- Adam lr={args.lr}, epochs={args.epochs}, orthogonal ID init
- Vary **only σ** ∈ {sigmas}
- Seed 0 at every σ; seeds {{0,1,2}} at σ ∈ {{0.05, 0.2}}

## Hypotheses (pre-registered before results)
1. {HYPOTHESES['i']}
2. {HYPOTHESES['ii']}
3. {HYPOTHESES['iii']}

## Primary metric (§5 — dense: rel-Frobenius A_cur vs ΛW) — seed 0

| σ | attn | raw ridge | proj ridge | Spearman | stub corr | attn MSE | κ | κ_Z | attn−ridge gap |
|---|------|-----------|------------|----------|-----------|----------|---|-----|----------------|
{chr(10).join(rows)}

## Multi-seed checks
{chr(10).join(ms_notes) if ms_notes else "(none)"}

## Ridge-baseline recovery
See raw / projected columns above (same data per σ).

## Prediction MSE
See attn MSE column; ridge MSE in `metrics.json` per run.

## State-covariance condition numbers
- Marginal κ and joint κ_Z both logged; plot: `recovery_and_kappa_vs_sigma.png`

## Heatmaps
Per-σ / per-seed under `sigma_*/seed_*/heatmap_A_vs_W.png`

## Verdicts on hypotheses
- {vline('i')}
- {vline('ii')}
- {vline('iii')}
"""
    (out / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
