"""Enhanced classical sampler network.

This module defines SamplerQNN, a deeper feed‑forward network with batch
normalisation, dropout and advanced weight‑initialisation.  It is useful
for benchmarking against the quantum counterpart and for hybrid training
pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """Classical sampler network with 2‑to‑4‑to‑8‑to‑4‑to‑2 architecture.

    The network maps a 2‑dimensional input to a 2‑dimensional probability
    vector.  Dropout and batch‑normalisation improve generalisation and
    training stability.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        # initialise weights to promote stable gradients
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return a probability distribution over two classes."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)


__all__ = ["SamplerQNN"]
