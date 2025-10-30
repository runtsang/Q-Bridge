#!/usr/bin/env python
"""Enhanced EstimatorQNN: a two‑layer MLP with dropout and batch‑norm."""
from __future__ import annotations

import torch
from torch import nn

class EstimatorNN(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

def EstimatorQNN(**kwargs) -> nn.Module:
    """Return a fully‑connected regression network with optional dropout and batch‑norm."""
    return EstimatorNN(**kwargs)

__all__ = ["EstimatorQNN"]
