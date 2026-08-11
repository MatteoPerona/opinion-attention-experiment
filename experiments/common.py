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
    row_softmax_normalize,
    spearman_corr_entries,
    topk_edge_precision,
)
from sim import state_covariance_condition_number


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
    W_hat_stoch = row_softmax_normalize(W_hat)
    return {
        "W_hat": W_hat,
        "W_hat_stoch": W_hat_stoch,
        "rel_frobenius_raw": relative_frobenius(W_hat, W),
        "rel_frobenius_stoch": relative_frobenius(W_hat_stoch, W),
        "spearman_raw": spearman_corr_entries(W_hat, W),
        "spearman_stoch": spearman_corr_entries(W_hat_stoch, W),
        "topk_precision_stoch": topk_edge_precision(W_hat_stoch, W),
        "pred_mse": float(np.mean((X @ W_hat.T - Y) ** 2)),
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
    target_cur = np.diag(world_lam) @ world_W
    target_anc = np.diag(1.0 - world_lam)
    Z = np.concatenate([X, x0], axis=1)
    pred = Z @ Op.T
    return {
        "Op": Op,
        "W_cur": W_cur,
        "W_anc": W_anc,
        "rel_frobenius_cur": relative_frobenius(W_cur, target_cur),
        "rel_frobenius_anc": relative_frobenius(W_anc, target_anc),
        "spearman_cur": spearman_corr_entries(W_cur, target_cur),
        "anchor_diag_mae": float(np.mean(np.abs(np.diag(W_anc) - (1.0 - world_lam)))),
        "pred_mse": float(np.mean((pred - Y) ** 2)),
    }


def condition_number_report(pairs_X: np.ndarray) -> float:
    return state_covariance_condition_number(pairs_X)
