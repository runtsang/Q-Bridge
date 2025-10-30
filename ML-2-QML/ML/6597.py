import torch
import torch.nn as nn
from typing import Sequence

class QCNNModel(nn.Module):
    """
    Fully‑connected network that mirrors the quantum convolution pattern.
    Allows dynamic layer sizes for flexibility.
    """

    def __init__(self, input_dim: int = 8, layer_sizes: Sequence[int] | None = None, out_features: int = 1):
        super().__init__()
        if layer_sizes is None:
            # Default: 8→16→12→8→4→4
            layer_sizes = [8, 16, 12, 8, 4, 4]
        layers = []
        prev = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.Tanh())
            prev = size
        layers.append(nn.Linear(prev, out_features))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))

def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
