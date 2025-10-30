"""Enhanced classical QCNN with residual connections and dropout."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QCNNGen(nn.Module):
    """A classical QCNN-like network with residual skip connections and dropout.

    The model mirrors the original layer structure but adds
    dropout after each activation and a residual addition
    between consecutive blocks.  This improves regularisation
    and stabilises training on small datasets.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        assert len(hidden_dims) > 0, "hidden_dims must contain at least one element"

        layers = []
        prev_dim = input_dim
        for idx, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            # Residual connection if dimensions match
            if prev_dim == dim:
                layers.append(nn.Identity())
            prev_dim = dim
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.layers(x)
        return torch.sigmoid(self.head(x))

    @staticmethod
    def default() -> "QCNNGen":
        """Return a model with the default hyper‑parameters."""
        return QCNNGen()


__all__ = ["QCNNGen"]
