"""Enhanced feed‑forward regressor with residuals, dropout and batch‑norm.

The model accepts 2‑dimensional input and produces a scalar output.
It is deliberately more expressive than the original tiny network,
allowing easier transfer to higher‑dimensional settings while still
remaining lightweight enough for quick experimentation.
"""

from __future__ import annotations

import torch
from torch import nn


def EstimatorQNN() -> nn.Module:
    """Return an extended regression network.

    The architecture:
      - Input → Linear(2, 8) → BatchNorm1d → ReLU
      - Linear(8, 4) → BatchNorm1d → ReLU
      - Linear(4, 1)
      - Dropout(0.1) applied after each hidden layer
      - Residual connection from input to final output (if sizes match)
    """
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(2, 8)
            self.bn1 = nn.BatchNorm1d(8)
            self.fc2 = nn.Linear(8, 4)
            self.bn2 = nn.BatchNorm1d(4)
            self.fc3 = nn.Linear(4, 1)
            self.dropout = nn.Dropout(0.1)
            self.activation = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Ensure input shape is (batch, 2)
            h = self.activation(self.bn1(self.fc1(x)))
            h = self.dropout(h)
            h = self.activation(self.bn2(self.fc2(h)))
            h = self.dropout(h)
            out = self.fc3(h)
            # Residual if possible
            if x.shape[-1] == out.shape[-1]:
                out = out + x[:, :1]
            return out

    return EstimatorNN()


__all__ = ["EstimatorQNN"]
