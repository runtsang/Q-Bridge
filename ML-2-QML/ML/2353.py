"""Hybrid self-attention combining classical and quantum-inspired scoring."""

from __future__ import annotations

import torch
from torch import nn


class HybridSelfAttention(nn.Module):
    """Classical self‑attention block that uses a small neural network to compute
    attention logits, mirroring the EstimatorQNN architecture.  The layer
    accepts a batch of sequences and returns the attention‑weighted
    representations.

    The design is inspired by the seed SelfAttention and EstimatorQNN modules:
    * Query/key/value projections from the seed SelfAttention.
    * A feed‑forward network with ``Tanh`` non‑linearity from EstimatorQNN
      to produce scalar scores for each query‑key pair.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Linear projections
        self.query_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        # Attention scoring network (EstimatorQNN style)
        self.attn_net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(batch, seq_len, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, seq_len, embed_dim)`` containing the
            attention‑weighted representations.
        """
        q = self.query_lin(inputs)
        k = self.key_lin(inputs)
        v = self.value_lin(inputs)

        batch, seq_len, _ = q.shape
        # Compute attention logits using the small neural network
        logits = torch.empty(batch, seq_len, seq_len, device=q.device)
        for i in range(seq_len):
            for j in range(seq_len):
                pair = torch.cat([q[:, i], k[:, j]], dim=-1)
                logits[:, i, j] = self.attn_net(pair).squeeze(-1)

        attn = torch.softmax(logits, dim=-1)
        return torch.einsum("bij,bjd->bid", attn, v)
