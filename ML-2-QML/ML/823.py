"""
EnhancedSamplerQNN – A richer classical sampler network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedSamplerQNN(nn.Module):
    """
    A two‑input, two‑output MLP with two hidden layers, batch‑norm, ReLU, and dropout.
    Provides convenient sampling and loss utilities for use in hybrid training loops.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int, int] = (8, 8),
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Compute a probability distribution over the two output classes.
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw discrete samples from the probability distribution returned by the network.
        """
        return torch.multinomial(probs, num_samples, replacement=True)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Cross‑entropy loss between logits and integer class labels.
        """
        return F.cross_entropy(logits, targets)

__all__ = ["EnhancedSamplerQNN"]
