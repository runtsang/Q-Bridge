"""
Deepened classical sampler with regularisation and probabilistic output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    A two‑hidden‑layer neural network that maps a 2‑dimensional input into a
    2‑dimensional probability distribution via softmax.  Dropout and
    batch‑normalisation are added to improve generalisation for downstream
    sampling tasks.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 2,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Compute the probability distribution over the two output classes.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (..., 2) containing the raw input features.

        Returns
        -------
        torch.Tensor
            Softmax‑normalised probability tensor of shape (..., 2).
        """
        return F.softmax(self.net(inputs), dim=-1)

__all__ = ["SamplerQNN"]
