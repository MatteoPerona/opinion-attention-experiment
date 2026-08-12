"""Shared experiment utilities: metrics, heatmaps, training loops, I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from baselines import (
    relative_frobenius,
    ridge_fit_FJ_operator,
    ridge_fit_W,
    project_rows_simplex,
    row_softmax_normalize,
    spearman_corr_entries,
    topk_edge_precision,
)
from sim import (
    joint_feature_covariance_condition_number,
    state_covariance_condition_number,
)

def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def save_heatmaps(
    A: np.ndarray,
    W: np.ndarray,
    out_path: Path,
    title_left: str = "Learned A",
    title_right: str = "True W",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmin = min(A.min(), W.min())
    vmax = max(A.max(), W.max())
    im0 = axes[0].imshow(A, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title(title_left)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(W, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title(title_right)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def degroot_metrics(A: np.ndarray, W: np.ndarray) -> dict:
    return {
        "rel_frobenius": relative_frobenius(A, W),
        "spearman": spearman_corr_entries(A, W),
        "topk_precision": topk_edge_precision(A, W),
    }


def train_degroot(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str = "cpu",
) -> list[float]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    P = X.shape[0]
    losses = []
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    for ep in range(epochs):
        perm = torch.randperm(P, device=device)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, P, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(Xt[idx])
            loss = torch.mean((pred - Yt[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        losses.append(ep_loss / max(n_batches, 1))
    return losses


def train_fj(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    x0: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str = "cpu",
    optimizer: str = "adam",
) -> list[float]:
    """
    Train FJ anchor-token head.
    optimizer: 'adam' (minibatch) or 'lbfgs' (full-batch; useful at N=50).
    """
    model = model.to(device)
    P = X.shape[0]
    losses = []
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    x0t = torch.tensor(x0, dtype=torch.float32, device=device)

    if optimizer == "lbfgs":
        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        for ep in range(epochs):
            def closure():
                opt.zero_grad()
                pred = model(Xt, x0t)
                loss = torch.mean((pred - Yt) ** 2)
                loss.backward()
                return loss

            loss = opt.step(closure)
            losses.append(float(loss.detach().item()))
        return losses

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(P, device=device)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, P, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(Xt[idx], x0t[idx])
            loss = torch.mean((pred - Yt[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        losses.append(ep_loss / max(n_batches, 1))
    return losses


@torch.no_grad()
def eval_mse_degroot(model: nn.Module, X: np.ndarray, Y: np.ndarray, device: str = "cpu") -> float:
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    pred = model(Xt)
    return float(torch.mean((pred - Yt) ** 2).item())


def train_fj_unrolled(
    model: nn.Module,
    trajectories: np.ndarray,
    x0s: np.ndarray,
    k: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str = "cpu",
    grad_clip: float = 1.0,
) -> list[float]:
    """
    k-step unrolled FJ training (§4 graduation path).

    Feed true x(t), roll the model k steps with its own predictions as current
    opinions; anchors stay fixed at true x(0). Loss = mean MSE over steps 1..k
    against stored true states. Only starts with t+k <= T are used.
    """
    model = model.to(device)
    # trajectories: (M, T+1, N)
    M, Tp1, N = trajectories.shape
    T = Tp1 - 1
    if k > T:
        raise ValueError(f"k={k} exceeds T={T}")
    # usable starts: t = 0..T-k  → for each traj, (T-k+1) starts
    starts = []
    for m in range(M):
        for t in range(T - k + 1):
            starts.append((m, t))
    starts = np.array(starts)
    P = len(starts)

    traj_t = torch.tensor(trajectories, dtype=torch.float32, device=device)
    x0_t = torch.tensor(x0s, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for ep in range(epochs):
        perm = np.random.permutation(P)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, P, batch_size):
            idx = starts[perm[start : start + batch_size]]
            m_idx = torch.tensor(idx[:, 0], device=device)
            t_idx = torch.tensor(idx[:, 1], device=device)
            x = traj_t[m_idx, t_idx]  # (B, N) true x(t)
            x0b = x0_t[m_idx]
            step_losses = []
            x_cur = x
            for s in range(1, k + 1):
                x_cur = model(x_cur, x0b)
                target = traj_t[m_idx, t_idx + s]
                step_losses.append(torch.mean((x_cur - target) ** 2))
            loss = torch.stack(step_losses).mean()
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        losses.append(ep_loss / max(n_batches, 1))
    return losses


@torch.no_grad()
def eval_mse_fj(
    model: nn.Module, X: np.ndarray, Y: np.ndarray, x0: np.ndarray, device: str = "cpu"
) -> float:
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    x0t = torch.tensor(x0, dtype=torch.float32, device=device)
    pred = model(Xt, x0t)
    return float(torch.mean((pred - Yt) ** 2).item())


def ridge_degroot_bundle(X: np.ndarray, Y: np.ndarray, W: np.ndarray, alpha: float = 1e-3) -> dict:
    W_hat = ridge_fit_W(X, Y, alpha=alpha)
    W_hat_relu = row_softmax_normalize(W_hat)
    W_hat_simplex = project_rows_simplex(W_hat)
    return {
        "W_hat": W_hat,
        "W_hat_stoch": W_hat_relu,
        "W_hat_simplex": W_hat_simplex,
        "rel_frobenius_raw": relative_frobenius(W_hat, W),
        "rel_frobenius_stoch": relative_frobenius(W_hat_relu, W),
        "rel_frobenius_simplex": relative_frobenius(W_hat_simplex, W),
        "spearman_raw": spearman_corr_entries(W_hat, W),
        "spearman_stoch": spearman_corr_entries(W_hat_relu, W),
        "spearman_simplex": spearman_corr_entries(W_hat_simplex, W),
        "topk_precision_stoch": topk_edge_precision(W_hat_relu, W),
        "pred_mse": float(np.mean((X @ W_hat.T - Y) ** 2)),
        "pred_mse_simplex": float(np.mean((X @ W_hat_simplex.T - Y) ** 2)),
    }


def ridge_fj_bundle(
    X: np.ndarray,
    Y: np.ndarray,
    x0: np.ndarray,
    world_W: np.ndarray,
    world_lam: np.ndarray,
    alpha: float = 1e-3,
) -> dict:
    Op = ridge_fit_FJ_operator(X, Y, x0, alpha=alpha)
    N = world_W.shape[0]
    W_cur = Op[:, :N]
    W_anc = Op[:, N:]
    Op_simplex = project_rows_simplex(Op)
    W_cur_s = Op_simplex[:, :N]
    W_anc_s = Op_simplex[:, N:]
    target_cur = np.diag(world_lam) @ world_W
    target_anc = np.diag(1.0 - world_lam)
    Z = np.concatenate([X, x0], axis=1)
    pred = Z @ Op.T
    pred_s = Z @ Op_simplex.T
    return {
        "Op": Op,
        "Op_simplex": Op_simplex,
        "W_cur": W_cur,
        "W_anc": W_anc,
        "rel_frobenius_cur": relative_frobenius(W_cur, target_cur),
        "rel_frobenius_anc": relative_frobenius(W_anc, target_anc),
        "rel_frobenius_cur_simplex": relative_frobenius(W_cur_s, target_cur),
        "rel_frobenius_anc_simplex": relative_frobenius(W_anc_s, target_anc),
        "spearman_cur": spearman_corr_entries(W_cur, target_cur),
        "spearman_cur_simplex": spearman_corr_entries(W_cur_s, target_cur),
        "anchor_diag_mae": float(np.mean(np.abs(np.diag(W_anc) - (1.0 - world_lam)))),
        "pred_mse": float(np.mean((pred - Y) ** 2)),
        "pred_mse_simplex": float(np.mean((pred_s - Y) ** 2)),
    }


def condition_number_report(pairs_X: np.ndarray) -> float:
    return state_covariance_condition_number(pairs_X)


def joint_condition_number_report(pairs_X: np.ndarray, pairs_x0: np.ndarray) -> float:
    return joint_feature_covariance_condition_number(pairs_X, pairs_x0)
