import torch
from torch import nn
import numpy as np
from typing import Sequence

class ConvFilter(nn.Module):
    """Classical convolutional filter with optional regularisation."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 bias: bool = True, batch_norm: bool = False,
                 dropout: float | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias)
        self.bn = nn.BatchNorm2d(1) if batch_norm else None
        self.dropout = nn.Dropout2d(dropout) if dropout else None

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = self.conv(data)
        if self.bn is not None:
            x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return torch.sigmoid(x - self.threshold)

class QuantumKernel(nn.Module):
    """Hybrid quantum kernel – classical surrogate for quantum evaluation."""
    def __init__(self, n_wires: int = 4, seed: int | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.params = nn.Parameter(torch.randn(n_wires))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = (x - y).abs()
        return torch.exp(-diff.sum())

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class ConvGen175(nn.Module):
    """Combined classical conv filter and quantum kernel."""
    def __init__(self,
                 kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 conv_bias: bool = True,
                 conv_batch_norm: bool = False,
                 conv_dropout: float | None = None,
                 kernel_gamma: float = 1.0) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size, conv_threshold,
                               bias=conv_bias,
                               batch_norm=conv_batch_norm,
                               dropout=conv_dropout)
        self.kernel = QuantumKernel(n_wires=4)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_feat = self.conv(x)
        y_feat = self.conv(y)
        return self.kernel(x_feat.flatten(1), y_feat.flatten(1))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b)

__all__ = ["ConvGen175", "ConvFilter", "QuantumKernel", "kernel_matrix"]
