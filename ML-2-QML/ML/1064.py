"""SamplerQNN: A classical sampler network with residual connections and dropout.

This class extends the original two‑layer MLP by adding batch normalization,
dropout, and a residual skip connection. It outputs a probability distribution
over two classes via a softmax layer. The network can be used as a standalone
module or as part of a larger hybrid model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """Classical sampler network with residual connections and dropout."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = x + residual
        logits = self.out(x)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNN"]
