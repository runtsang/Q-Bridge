"""Enhanced classical sampler network with training utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SamplerQNN(nn.Module):
    """
    A two‑input, two‑output neural sampler with modern regularisation.

    The network maps a 2‑D input vector to a probability distribution over
    two outcomes.  It contains:
      * Linear → BatchNorm1d → ReLU
      * Linear → Dropout
      * Linear → Softmax

    The forward method returns the probability vector.  A small helper
    ``sample`` method draws a one‑hot sample from the distribution.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.Dropout(dropout),
            nn.Linear(8, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Draw a one‑hot sample from the probability distribution.

        Parameters
        ----------
        probs : torch.Tensor
            Tensor of shape (..., 2) containing probabilities.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape with a single 1 in the chosen class.
        """
        batch_size = probs.shape[0]
        idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return F.one_hot(idx, num_classes=2).float()

__all__ = ["SamplerQNN"]
