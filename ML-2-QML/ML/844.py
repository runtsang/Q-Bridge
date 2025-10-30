"""Enhanced classical sampler network with deeper layers and regularisation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    A regularised, multi‑layer neural network that maps a 2‑dimensional input
    to a 2‑dimensional probability distribution.

    Architecture:
        Input (2) -> Linear(2, 4) -> BatchNorm1d(4) -> ReLU
        -> Dropout(0.2) -> Linear(4, 8) -> BatchNorm1d(8) -> ReLU
        -> Dropout(0.2) -> Linear(8, 2) -> Softmax
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)


__all__ = ["SamplerQNN"]
