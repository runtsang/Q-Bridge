"""Hybrid QCNN combining classical convolution and quantum-inspired layers."""
from __future__ import annotations

import torch
from torch import nn

class ConvFilter(nn.Module):
    """Classical convolution filter emulating a quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

class QCNNModel(nn.Module):
    """Classical QCNN model mirroring quantum convolution steps."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridQCNN(nn.Module):
    """Hybrid classical‑quantum convolutional network."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.pre = ConvFilter(kernel_size, threshold)
        self.flatten = nn.Flatten()
        self.model = QCNNModel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch, 1, 8, 8)
        x = self.pre(x)          # (batch, 1, 8, 8)
        x = self.flatten(x)      # (batch, 64)
        x = self.model(x)        # (batch, 1)
        return x

def HybridQCNNFactory() -> HybridQCNN:
    """Convenience factory returning a configured `HybridQCNN`."""
    return HybridQCNN()

__all__ = ["HybridQCNN", "HybridQCNNFactory"]
