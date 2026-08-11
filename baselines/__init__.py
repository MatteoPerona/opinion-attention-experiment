"""Ridge-regression baseline for influence-matrix recovery."""

from __future__ import annotations

import numpy as np


def ridge_fit_W(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 1e-3,
) -> np.ndarray:
    """
    Fit Y ≈ W @ X.T style: each column of Y^T ≈ W @ corresponding column of X^T.

    We solve row-wise / globally:
      W_hat = Y.T @ X @ inv(X.T @ X + alpha I)
    so that Y ≈ X @ W.T  with W approximating the left-multiply operator
    x+ = W x  ⇒  stacking rows: Y = X @ W.T.

    X, Y: (P, N) arrays of states.
    Returns W_hat of shape (N, N).
    """
    P, N = X.shape
    assert Y.shape == (P, N)
    XtX = X.T @ X + alpha * np.eye(N)
    # Y = X @ W.T  ⇒  W.T = solve(XtX, X.T @ Y)  ⇒  W = (X.T @ Y).T @ inv... wait
    # Actually: W.T = (X.T X)^{-1} X.T Y  ⇒ W = Y.T X (X.T X)^{-1}
    W = (Y.T @ X) @ np.linalg.inv(XtX)
    return W


def ridge_fit_FJ_operator(
    X: np.ndarray,
    Y: np.ndarray,
    x0: np.ndarray,
    alpha: float = 1e-3,
) -> np.ndarray:
    """
    Fit Y ≈ [W_cur | W_anc] @ [X; x0]  i.e. an N x 2N operator.

    Returns Op of shape (N, 2N): Op[:, :N] ≈ Lambda W, Op[:, N:] ≈ (I - Lambda)
    (ideally diagonal on the anchor block).
    """
    P, N = X.shape
    assert Y.shape == (P, N) and x0.shape == (P, N)
    Z = np.concatenate([X, x0], axis=1)  # (P, 2N)
    ZtZ = Z.T @ Z + alpha * np.eye(2 * N)
    # Y = Z @ Op.T  ⇒ Op = Y.T @ Z @ inv(Z.T Z)
    Op = (Y.T @ Z) @ np.linalg.inv(ZtZ)
    return Op


def relative_frobenius(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, "fro") / (np.linalg.norm(B, "fro") + 1e-15))


def spearman_corr_entries(A: np.ndarray, B: np.ndarray) -> float:
    """Spearman rank correlation of flattened matrix entries."""
    from scipy.stats import spearmanr

    r, _ = spearmanr(A.ravel(), B.ravel())
    return float(r)


def topk_edge_precision(A: np.ndarray, W_true: np.ndarray, k: int | None = None) -> float:
    """
    Precision@k where k = number of true positive edges (W_true > threshold).
    Predicted edges = top-k entries of A (excluding optional zeros in true).
    """
    true_mask = W_true > 1e-8
    if k is None:
        k = int(true_mask.sum())
    k = max(k, 1)
    # rank predicted
    flat = A.ravel()
    top_idx = np.argpartition(flat, -k)[-k:]
    pred_mask = np.zeros_like(flat, dtype=bool)
    pred_mask[top_idx] = True
    pred_mask = pred_mask.reshape(A.shape)
    tp = np.logical_and(pred_mask, true_mask).sum()
    return float(tp / k)


def row_softmax_normalize(A: np.ndarray) -> np.ndarray:
    """Project a matrix to row-stochastic via ReLU+normalize (for fair comparison)."""
    A = np.maximum(A, 0)
    s = A.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return A / s
