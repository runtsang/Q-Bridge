"""Enhanced classical self‑attention with trainable projections and temperature.

This module extends the original seed by adding separate linear
projections for Q, K, V and a learnable temperature parameter.
The class exposes a `forward` method that can be used in a
neural‑network pipeline and a simple `train_step` helper for quick
experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class SelfAttentionEnhanced(nn.Module):
    """
    Classical self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    temperature : float, optional
        Initial temperature for the softmax.  Can be learned.
    """

    def __init__(
        self,
        embed_dim: int,
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Separate projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute self‑attention for a batch of sequences.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, seq_len, embed_dim).
        """
        Q = self.q_proj(inputs)  # (B, L, E)
        K = self.k_proj(inputs)  # (B, L, E)
        V = self.v_proj(inputs)  # (B, L, E)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.embed_dim**0.5) * self.temperature
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module = nn.MSELoss(),
    ) -> float:
        """
        Perform one gradient‑descent step.

        Parameters
        ----------
        inputs : torch.Tensor
            Input batch.
        targets : torch.Tensor
            Target outputs.
        optimizer : torch.optim.Optimizer
            Optimizer instance.
        loss_fn : nn.Module, optional
            Loss function.

        Returns
        -------
        float
            Loss value for monitoring.
        """
        self.train()
        optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()


__all__ = ["SelfAttentionEnhanced"]
