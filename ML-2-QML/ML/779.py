"""Enhanced classical sampler network with deeper architecture and regularization.

The network accepts a 2‑dimensional input and produces a probability
distribution over two classes.  Compared to the seed implementation
it adds batch‑normalisation, dropout and an additional hidden layer,
making it more expressive while still being lightweight.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen(nn.Module):
    """Two‑hidden‑layer MLP with batch‑norm and dropout."""
    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int, int] = (8, 4), output_dim: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Draw samples from the categorical distribution defined by the network."""
        probs = self.forward(x)
        return torch.multinomial(probs, num_samples=num_samples, replacement=True)

    def get_params(self):
        """Return a list of trainable parameters."""
        return list(self.parameters())
