"""Single attention head: identity-driven Q/K, raw-scalar values."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityAttentionHead(nn.Module):
    """
    DeGroot-style single head.

    - Learnable per-agent identity embeddings e_i in R^d, d >= N
    - q_i = W_Q e_i, k_j = W_K e_j  (identity only — not opinion)
    - A_ij = softmax_j(q_i · k_j / sqrt(d))
    - xhat_i = sum_j A_ij * x_j   (raw scalar value path; no W_V, no out-proj, no MLP)

    Forward input: x of shape (batch, N) or (N,)
    """

    def __init__(self, N: int, d: int | None = None):
        super().__init__()
        if d is None:
            d = N
        if d < N:
            raise ValueError(f"Capacity requirement: d >= N (got d={d}, N={N})")
        self.N = N
        self.d = d
        self.id_embed = nn.Embedding(N, d)
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.id_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)

    def attention_matrix(self) -> torch.Tensor:
        """Return the N x N attention map A (row-stochastic), independent of opinions."""
        ids = torch.arange(self.N, device=self.id_embed.weight.device)
        e = self.id_embed(ids)  # (N, d)
        q = self.W_Q(e)
        k = self.W_K(e)
        logits = (q @ k.T) / (self.d**0.5)
        return F.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., N) opinions
        returns: (..., N) predicted next opinions
        """
        A = self.attention_matrix()  # (N, N)
        # xhat = A @ x  for each batch row
        return torch.matmul(x, A.T)


class FJAnchorAttentionHead(nn.Module):
    """
    FJ single head with 2N tokens: N current + N anchor.

    Values are raw scalars for both token types.
    Agent i's prediction = attention-weighted sum over all 2N token values.
    Learned map should put ~lambda_i W_ij on current tokens and ~(1-lambda_i)
    on agent i's own anchor token.
    """

    def __init__(self, N: int, d: int | None = None):
        super().__init__()
        if d is None:
            d = N
        if d < N:
            raise ValueError(f"Capacity requirement: d >= N (got d={d}, N={N})")
        self.N = N
        self.d = d
        # 2N identity embeddings: [0..N) current agents, [N..2N) anchors
        self.id_embed = nn.Embedding(2 * N, d)
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        # Only current-opinion agents produce outputs (queries from first N)
        nn.init.normal_(self.id_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)

    def attention_matrix(self) -> torch.Tensor:
        """
        Return (N, 2N) attention: rows = current agents, cols = [current | anchors].
        """
        device = self.id_embed.weight.device
        cur_ids = torch.arange(self.N, device=device)
        all_ids = torch.arange(2 * self.N, device=device)
        e_q = self.id_embed(cur_ids)
        e_k = self.id_embed(all_ids)
        q = self.W_Q(e_q)
        k = self.W_K(e_k)
        logits = (q @ k.T) / (self.d**0.5)
        return F.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """
        x, x0: (..., N)
        returns: (..., N)
        """
        A = self.attention_matrix()  # (N, 2N)
        # values = concat(x, x0) along last dim → (..., 2N)
        vals = torch.cat([x, x0], dim=-1)
        # xhat_i = sum_j A_ij vals_j
        return torch.matmul(vals, A.T)

    def current_block(self) -> torch.Tensor:
        return self.attention_matrix()[:, : self.N]

    def anchor_block(self) -> torch.Tensor:
        return self.attention_matrix()[:, self.N :]
