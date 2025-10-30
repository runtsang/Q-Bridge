"""
Class SamplerQNNGen072: Extended classical sampler network.

Features:
- Two hidden layers with dropout.
- Residual connections for improved gradient flow.
- Softmax output for probability distribution over two classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen072(nn.Module):
    def __init__(self, hidden_dim: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # First hidden layer with dropout
        x = self.dropout(F.tanh(self.fc1(inputs)))
        skip = x  # Residual skip connection

        # Second hidden layer with dropout
        x = self.dropout(F.tanh(self.fc2(x)))

        # Add residual connection
        x = x + skip

        # Output layer
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

__all__ = ["SamplerQNNGen072"]
