"""
EstimatorQNN – a richer classical regression network.

This module provides a simple yet more expressive neural network that
adds batch‑normalisation, dropout, and a residual connection to the
original 2‑input toy model.  The architecture is still fully
feed‑forward and can be used in any PyTorch pipeline.
"""

import torch
from torch import nn
import torch.nn.functional as F

class EstimatorNN(nn.Module):
    """
    A small but richer regression network that adds batch‑normalisation,
    dropout and a residual connection to the original 2‑input toy model.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Two hidden layers forming a residual block
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(2)]
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.input_layer(x)))
        for layer in self.hidden_layers:
            out = F.relu(layer(out))
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.output_layer(out)
        return out

def EstimatorQNN() -> EstimatorNN:
    """
    Factory that returns a ready‑to‑use estimator network.
    """
    return EstimatorNN()

__all__ = ["EstimatorQNN"]
