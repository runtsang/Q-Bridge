"""Hybrid classical kernel and classifier implementation.

This module unifies the classical RBF kernel utilities and a feed‑forward
classifier factory, exposing a single ``HybridKernelClassifier`` class
that can be instantiated with a kernel and a classifier.  The design
mirrors the quantum interface used in the QML counterpart, enabling
side‑by‑side experimentation and benchmarking.

The class provides:
- ``fit``: train the classifier on kernel features.
- ``predict``: predict labels for new data.
- ``kernel_matrix``: compute the Gram matrix using the chosen kernel.
- ``build_classifier_circuit``: returns a PyTorch ``nn.Sequential`` model
  along with metadata (encoding indices, weight sizes, observables).

The implementation relies only on NumPy and PyTorch, ensuring
compatibility with any classical pipeline.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence

import numpy as np
import torch
from torch import nn
from torch import optim


class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """RBF kernel module that wraps :class:`KernalAnsatz`."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between datasets ``a`` and ``b`` using an RBF kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a feed‑forward classifier and metadata similar to the quantum variant."""
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


class HybridKernelClassifier(nn.Module):
    """Hybrid kernel + classifier for classical experiments.

    The class encapsulates a kernel (default: RBF) and a feed‑forward
    classifier.  It exposes ``fit`` and ``predict`` methods that
    operate on kernel features, mirroring the quantum interface.
    """

    def __init__(self, kernel: Kernel, classifier: nn.Module, epochs: int = 100, lr: float = 1e-3) -> None:
        super().__init__()
        self.kernel = kernel
        self.classifier = classifier
        self.epochs = epochs
        self.lr = lr
        self.X_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Train the classifier on kernel features."""
        self.X_train = X
        self.y_train = y
        K = torch.tensor(kernel_matrix(X, X, gamma=self.kernel.ansatz.gamma), dtype=torch.float32)
        optimizer = optim.Adam(self.classifier.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.classifier(K)
            loss = criterion(logits, y.long())
            loss.backward()
            optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict labels for new data."""
        if self.X_train is None:
            raise RuntimeError("Model has not been fitted yet.")
        K = torch.tensor(kernel_matrix(X, self.X_train, gamma=self.kernel.ansatz.gamma), dtype=torch.float32)
        with torch.no_grad():
            logits = self.classifier(K)
        return torch.argmax(logits, dim=1)

    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Return the kernel matrix as a torch tensor."""
        return torch.tensor(kernel_matrix(X, Y, gamma=self.kernel.ansatz.gamma), dtype=torch.float32)


__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "build_classifier_circuit",
    "HybridKernelClassifier",
]
