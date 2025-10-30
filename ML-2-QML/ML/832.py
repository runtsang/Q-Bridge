"""Enhanced fully connected layer with optional hidden units, dropout and batch normalization."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable

class FCL(nn.Module):
    """A flexible fully‑connected neural layer supporting hidden units, dropout and batch‑norm.

    The original seed exposed a simple linear layer with a ``run`` method.  This
    extension adds a hidden representation, optional regularisation and a
    ``forward`` method that can be used in standard PyTorch pipelines.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim or n_features
        self.linear1 = nn.Linear(n_features, self.hidden_dim)
        self.bn = nn.BatchNorm1d(self.hidden_dim) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, thetas: Iterable[float] | torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning the mean activation."""
        x = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        x = self.linear1(x)
        x = self.bn(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return torch.tanh(x).mean(dim=0)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """Compatibility wrapper returning a NumPy array."""
        return self.forward(thetas).detach().numpy()

__all__ = ["FCL"]
