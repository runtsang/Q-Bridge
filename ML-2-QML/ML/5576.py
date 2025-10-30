"""Combined classical QCNN with convolutional filter and quantum-inspired fully connected layer."""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable
import numpy as np


class ConvFilter(nn.Module):
    """Emulates a quantum convolutional filter with a small 2×2 kernel."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


class FullyConnectedLayer(nn.Module):
    """A tiny fully‑connected module that mimics a quantum fully‑connected layer."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


class QCNNGen483(nn.Module):
    """Hybrid classical QCNN that chains feature extraction, convolution, pooling,
    a quantum‑style filter and a fully‑connected projection."""
    def __init__(self) -> None:
        super().__init__()
        # Feature extraction
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        # Convolution‑pooling stack
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Quantum‑style modules
        self.conv_filter = ConvFilter(kernel_size=2, threshold=0.0)
        self.fcl = FullyConnectedLayer(n_features=4)
        # Output head
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Simulate quantum filter on a reshaped tensor
        filter_output = self.conv_filter.run(x.detach().cpu().numpy().reshape(2, 2))
        # Simulate quantum fully‑connected layer
        fcl_output = self.fcl.run([filter_output] * 4)
        out = torch.tensor(fcl_output, dtype=torch.float32, device=inputs.device)
        return torch.sigmoid(self.head(out))


__all__ = ["QCNNGen483"]
