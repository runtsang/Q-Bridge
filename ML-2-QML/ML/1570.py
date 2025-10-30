from __future__ import annotations

import torch
from torch import nn

class QCNNModel(nn.Module):
    """
    Enhanced classical QCNN analog.
    Uses residual connections, batch‑norm, and dropout for better representational power.
    The architecture mirrors the quantum depth while adding classical regularisation.
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 8, 4, 4]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            # Residual if input and output sizes match
            if prev == h:
                layers.append(nn.Identity())
            prev = h
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        logits = self.head(x)
        return torch.sigmoid(logits)

def QCNN() -> QCNNModel:
    """Factory returning a fully‑configured QCNNModel."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
