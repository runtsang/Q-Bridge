"""Classical counterpart of QuanvolutionHybrid."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Deterministic 2×2 filter inspired by the quantum quanvolution."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)


class QuantumExpectationHead(nn.Module):
    """Dense head with a learned shift, mimicking a quantum expectation."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)


class QuanvolutionHybrid(nn.Module):
    """Classical quanvolution filter followed by a dense expectation head."""
    def __init__(self, in_channels: int = 1, n_qubits: int = 4) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter(in_channels, out_channels=n_qubits)
        self.head = QuantumExpectationHead(in_features=n_qubits * 14 * 14)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        probs = self.head(features)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuanvolutionFilter", "QuantumExpectationHead", "QuanvolutionHybrid"]
