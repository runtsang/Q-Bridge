"""
Samplers for quantum‑classical experiments.

This module defines `SamplerQNNGen222`, a deeper classical neural network
with batch normalisation and dropout.  It mirrors the structure of the
original two‑layer sampler but offers better regularisation and a more
flexible architecture for downstream hybrid workflows.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNNGen222(nn.Module):
    """
    A 3‑layer feed‑forward sampler with batch normalisation and dropout.

    Architecture:
        Input (2) → Linear(2→8) → ReLU → BatchNorm1d(8) → Dropout(0.2)
        → Linear(8→4) → ReLU → Linear(4→2) → Softmax.
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(8),
            nn.Dropout(dropout),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing a probability distribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (..., 2) with raw input features.

        Returns
        -------
        torch.Tensor
            Tensor of shape (..., 2) with class probabilities.
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)


__all__ = ["SamplerQNNGen222"]
