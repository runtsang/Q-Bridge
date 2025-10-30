"""HybridClassifier: classical implementation with kernel and conv support."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RBFKernel(nn.Module):
    """Radial basis function kernel with trainable gamma."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-gamma * ||x-y||^2)."""
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-self.gamma * (diff**2).sum(dim=-1))


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two sequences of tensors."""
    k = RBFKernel(gamma)
    return np.array([[k(x, y).item() for y in b] for x in a])


class FullyConnectedLayer(nn.Module):
    """Imitates the quantum FCL example but in pure PyTorch."""

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        output = torch.tanh(self.linear(values)).mean(dim=0)
        return output.detach().numpy()


class ConvFeatureExtractor(nn.Module):
    """Classical CNN that mirrors the QFCModel feature extractor."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.features(x))


class HybridClassifierML(nn.Module):
    """Flexible classical classifier that can prepend kernels or convs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int = 2,
        use_kernel: bool = False,
        kernel_gamma: float = 1.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        self.use_conv = use_conv

        if self.use_conv:
            self.conv = ConvFeatureExtractor()
            feature_dim = 16 * 7 * 7  # output size after two pooling layers
        else:
            feature_dim = input_dim

        layers: List[nn.Module] = []
        in_dim = feature_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)

        if self.use_kernel:
            self.kernel = RBFKernel(gamma=kernel_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            x = self.conv(x)
        if self.use_kernel:
            # For simplicity, we apply the kernel to the raw input, not to batches
            # This example is only illustrative: a true kernel classifier would
            # use a support‑vector machine or similar.
            raise NotImplementedError("Kernel‑preprocessing must be applied outside the forward pass.")
        return self.network(x)

__all__ = [
    "HybridClassifierML",
    "RBFKernel",
    "kernel_matrix",
    "FullyConnectedLayer",
    "ConvFeatureExtractor",
]
