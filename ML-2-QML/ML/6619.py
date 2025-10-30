"""Hybrid kernel‑classifier combining classical RBF and a neural net."""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn


class RBFKernel(nn.Module):
    """Classical radial basis function kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class HybridKernelClassifier:
    """Hybrid kernel‑classifier with classical RBF kernel and a torch network."""

    def __init__(self, gamma: float = 1.0, num_features: int = 10, depth: int = 2) -> None:
        self.kernel = RBFKernel(gamma)
        self.network, self.encoding, self.weight_sizes, self.observables = self._build_classifier(
            num_features, depth
        )

    def _build_classifier(self, num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
        """Create a feed‑forward network mimicking the quantum circuit interface."""
        layers: list[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: list[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix using the classical RBF kernel."""
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier network."""
        return self.network(x)


__all__ = ["HybridKernelClassifier"]
