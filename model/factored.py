"""Factored-logit FJ head — DIAGNOSTIC ONLY (outside run-#1 constraint set).

Agent i's row = (1 - g_i) * softmax_current(i, :)  concatenated with
g_i on own anchor and 0 on other anchors. Gate g_i ≈ 1-λ_i.
Isolates whether joint 2N-softmax is the rung-3 bottleneck.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactoredFJAttentionHead(nn.Module):
    """
    Architecture ablation (not a headline result).

    - Identity-driven Q/K over N current tokens only (as DeGroot head)
    - Learnable per-agent gate g_i = sigmoid(raw_i) ∈ (0,1) = mass on own anchor
    - Row: [(1-g_i) * softmax(q_i·k_j),  zeros..., g_i at anchor i, ...]
    - Values still raw scalars; no W_V / out-proj / MLP
    """

    def __init__(self, N: int, d: int | None = None):
        super().__init__()
        if d is None:
            d = N
        if d < N:
            raise ValueError(f"d >= N required (got d={d}, N={N})")
        self.N = N
        self.d = d
        self.id_embed = nn.Embedding(N, d)
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        # gate logits; init near 0.5
        self.gate_logit = nn.Parameter(torch.zeros(N))
        nn.init.orthogonal_(self.id_embed.weight)
        self.id_embed.weight.data *= 0.5
        nn.init.eye_(self.W_Q.weight)
        nn.init.eye_(self.W_K.weight)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def attention_matrix(self) -> torch.Tensor:
        """Return (N, 2N) row-stochastic factored attention."""
        device = self.id_embed.weight.device
        ids = torch.arange(self.N, device=device)
        e = self.id_embed(ids)
        q = self.W_Q(e)
        k = self.W_K(e)
        logits = (q @ k.T) / (self.d**0.5)
        A_cur = F.softmax(logits, dim=-1)  # (N, N)
        g = self.gates()  # (N,)
        A_cur_scaled = A_cur * (1.0 - g)[:, None]
        A_anc = torch.diag(g)
        return torch.cat([A_cur_scaled, A_anc], dim=-1)

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        A = self.attention_matrix()
        vals = torch.cat([x, x0], dim=-1)
        return torch.matmul(vals, A.T)
