"""Enhanced feed‑forward regressor with skip connections and dropout.

This module defines EstimatorQNN, a deeper and more regularised
regression network that builds on the original two‑layer example.
"""

import torch
from torch import nn
from torch.nn import functional as F

class EstimatorQNN(nn.Module):
    """
    A small but richer fully‑connected network.

    Architecture:
    - Input (2) → Linear(8) → BatchNorm1d → ReLU
    - Linear(16) → Dropout(0.2) → ReLU
    - Linear(8) → BatchNorm1d → ReLU
    - Linear(1)
    The skip connection from the first hidden layer to the third
    improves gradient flow for this toy problem.
    """
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 8, bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, 16, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(16, 8, bias=False)
        self.bn3 = nn.BatchNorm1d(8)
        self.out = nn.Linear(8, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.bn1(self.fc1(x)))
        h2 = F.relu(self.dropout(self.fc2(h1)))
        h3 = F.relu(self.bn3(self.fc3(h2)))
        # skip connection
        h3 = h3 + h1
        out = self.out(h3)
        return out

__all__ = ["EstimatorQNN"]
