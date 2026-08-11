"""Friedkin–Johnsen / DeGroot opinion dynamics simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


GraphKind = Literal["dense", "clustered", "sparse"]


@dataclass
class World:
    """Fixed (W, Lambda) for single-world system identification."""

    W: np.ndarray  # (N, N) row-stochastic
    Lambda: np.ndarray  # (N,) susceptibilities on the diagonal
    kind: GraphKind = "dense"

    @property
    def N(self) -> int:
        return self.W.shape[0]

    @property
    def Lambda_mat(self) -> np.ndarray:
        return np.diag(self.Lambda)


def sample_dense_W(N: int, rng: np.random.Generator) -> np.ndarray:
    """Dense random row-stochastic W: i.i.d. positive entries, row-normalized."""
    raw = rng.random((N, N)) + 1e-6
    return raw / raw.sum(axis=1, keepdims=True)


def sample_clustered_W(
    N: int,
    rng: np.random.Generator,
    n_clusters: int = 5,
    p_in: float = 0.8,
    p_out: float = 0.05,
    in_weight: float = 5.0,
    out_weight: float = 0.2,
) -> np.ndarray:
    """
    Sparse/clustered row-stochastic W: strong within-cluster, weak between.
    Edge presence Bernoulli; weights then row-normalized.
    """
    labels = np.array([i * n_clusters // N for i in range(N)])
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                w = in_weight if labels[i] == labels[j] else out_weight
                W[i, j] = w * (0.5 + rng.random())
        if W[i].sum() == 0:
            # ensure row has at least a self-loop
            W[i, i] = 1.0
    return W / W.sum(axis=1, keepdims=True)


def sample_sparse_W(
    N: int,
    rng: np.random.Generator,
    degree: int = 5,
) -> np.ndarray:
    """Erdős–Rényi-ish sparse row-stochastic W with roughly `degree` out-neighbors."""
    W = np.zeros((N, N))
    for i in range(N):
        k = min(degree, N)
        js = rng.choice(N, size=k, replace=False)
        W[i, js] = rng.random(k) + 1e-6
    return W / W.sum(axis=1, keepdims=True)


def make_world(
    N: int,
    rng: np.random.Generator,
    kind: GraphKind = "dense",
    susceptibilities: np.ndarray | None = None,
    degroot: bool = True,
    **graph_kwargs,
) -> World:
    if kind == "dense":
        W = sample_dense_W(N, rng)
    elif kind == "clustered":
        W = sample_clustered_W(N, rng, **graph_kwargs)
    elif kind == "sparse":
        W = sample_sparse_W(N, rng, **graph_kwargs)
    else:
        raise ValueError(f"Unknown graph kind: {kind}")

    if susceptibilities is not None:
        lam = np.asarray(susceptibilities, dtype=float)
    elif degroot:
        lam = np.ones(N)
    else:
        # FJ: all strictly < 1 so closed form is valid
        lam = rng.uniform(0.2, 0.9, size=N)

    return World(W=W, Lambda=lam, kind=kind)


def fj_step(
    x: np.ndarray,
    x0: np.ndarray,
    world: World,
    sigma: float,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """One FJ (or DeGroot) step with optional process noise."""
    lam = world.Lambda
    nxt = lam * (world.W @ x) + (1.0 - lam) * x0
    if sigma > 0:
        assert rng is not None
        nxt = nxt + rng.normal(0.0, sigma, size=x.shape)
    return nxt


def fj_steady_state(x0: np.ndarray, world: World) -> np.ndarray:
    """
    Closed-form FJ steady state. Valid only when all lambda_i < 1
    (so rho(Lambda W) < 1). Must NOT be used when Lambda = I (DeGroot).
    """
    if np.any(world.Lambda >= 1.0 - 1e-12):
        raise ValueError(
            "FJ closed-form steady state is invalid when any lambda_i >= 1 "
            "(DeGroot / singular case). Use empirical end-of-trajectory instead."
        )
    N = world.N
    I = np.eye(N)
    Lam = world.Lambda_mat
    A = I - Lam @ world.W
    b = (I - Lam) @ x0
    return np.linalg.solve(A, b)


def simulate_trajectory(
    x0: np.ndarray,
    world: World,
    T: int,
    sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate full trajectory of length T+1 (states x(0)..x(T)).
    Returns array of shape (T+1, N).
    """
    if sigma > 0 and rng is None:
        raise ValueError("rng required when sigma > 0")
    xs = [np.asarray(x0, dtype=float).copy()]
    x = xs[0]
    for _ in range(T):
        x = fj_step(x, xs[0], world, sigma, rng)
        xs.append(x)
    return np.stack(xs, axis=0)


def generate_dataset(
    world: World,
    M: int,
    T: int,
    sigma: float,
    rng: np.random.Generator,
    x0_scale: float = 1.0,
) -> dict:
    """
    One world, M trajectories from varied x(0).

    Returns dict with:
      trajectories: (M, T+1, N)
      x0s: (M, N)
      pairs_X: (M*T, N)  # x(t)
      pairs_Y: (M*T, N)  # x(t+1)
      pairs_x0: (M*T, N)  # innate opinion for that trajectory
      steady_empirical: (M, N)  # x(T)
      steady_closed: (M, N) or None  # closed form if valid
    """
    N = world.N
    trajs = np.zeros((M, T + 1, N))
    x0s = rng.normal(0.0, x0_scale, size=(M, N))
    use_closed = bool(np.all(world.Lambda < 1.0 - 1e-12))
    steady_closed = np.zeros((M, N)) if use_closed else None

    for m in range(M):
        trajs[m] = simulate_trajectory(x0s[m], world, T, sigma=sigma, rng=rng)
        if use_closed:
            steady_closed[m] = fj_steady_state(x0s[m], world)

    # Flatten consecutive pairs
    pairs_X = trajs[:, :-1, :].reshape(M * T, N)
    pairs_Y = trajs[:, 1:, :].reshape(M * T, N)
    pairs_x0 = np.repeat(x0s, T, axis=0)

    return {
        "trajectories": trajs,
        "x0s": x0s,
        "pairs_X": pairs_X,
        "pairs_Y": pairs_Y,
        "pairs_x0": pairs_x0,
        "steady_empirical": trajs[:, -1, :],
        "steady_closed": steady_closed,
        "world": world,
        "M": M,
        "T": T,
        "sigma": sigma,
    }


def state_covariance_condition_number(pairs_X: np.ndarray) -> float:
    """Condition number of empirical covariance of stacked states (N x N)."""
    X = pairs_X - pairs_X.mean(axis=0, keepdims=True)
    # cov over feature dim: states are rows
    cov = (X.T @ X) / max(X.shape[0] - 1, 1)
    # numerical floor for near-singular
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 1e-15, None)
    return float(eig.max() / eig.min())


def train_val_split(
    dataset: dict,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    """Split by trajectory index (held-out trajectories, not shuffled pairs)."""
    M = dataset["M"]
    T = dataset["T"]
    n_val = max(1, int(round(M * val_fraction)))
    n_train = M - n_val
    perm = rng.permutation(M)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    def _subset(idxs: np.ndarray) -> dict:
        trajs = dataset["trajectories"][idxs]
        x0s = dataset["x0s"][idxs]
        m = len(idxs)
        return {
            "trajectories": trajs,
            "x0s": x0s,
            "pairs_X": trajs[:, :-1, :].reshape(m * T, dataset["world"].N),
            "pairs_Y": trajs[:, 1:, :].reshape(m * T, dataset["world"].N),
            "pairs_x0": np.repeat(x0s, T, axis=0),
            "steady_empirical": trajs[:, -1, :],
            "steady_closed": (
                dataset["steady_closed"][idxs]
                if dataset["steady_closed"] is not None
                else None
            ),
            "world": dataset["world"],
            "M": m,
            "T": T,
            "sigma": dataset["sigma"],
            "traj_indices": idxs,
        }

    return _subset(train_idx), _subset(val_idx)
