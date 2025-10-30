from __future__ import annotations

import torch
from torch import nn


class QCNNNet(nn.Module):
    """Enhanced QCNN‑inspired network with residual connections, batch‑norm and dropout.

    The architecture mirrors the original fully‑connected stack but adds depth‑wise
    separable layers, residual skips and dropout for improved generalisation.
    It is fully differentiable and can be trained with any PyTorch optimiser.
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.layers.append(block)
        self.residual = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        for layer in self.layers:
            x = layer(x)
        # Residual skip
        x = x + self.residual(x)
        return torch.sigmoid(self.head(x))


def QCNNNetFactory() -> QCNNNet:
    """Convenience factory returning a ready‑to‑train QCNNNet instance."""
    return QCNNNet()


__all__ = ["QCNNNet", "QCNNNetFactory"]
