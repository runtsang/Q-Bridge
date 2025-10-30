"""Hybrid classical classifier combining an RBF kernel and a feed‑forward network.

This module implements a two‑stage pipeline:
1.  A classical RBF kernel expands the input into a feature vector representing
    similarities to a set of support vectors.
2.  A simple feed‑forward network maps this feature space to class logits.

The design mirrors the quantum counterpart but uses only NumPy / PyTorch
operations, making it suitable for environments without a quantum backend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple

# ---------- Classical kernel -------------------------------------------------
class KernalAnsatz(nn.Module):
    """RBF kernel ansatz compatible with the quantum interface."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wraps :class:`KernalAnsatz` to provide a convenient API."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ---------- Classical classifier ---------------------------------------------
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a feed‑forward network with ``depth`` hidden layers."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
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

# ---------- Hybrid model ------------------------------------------------------
class SharedClassName(nn.Module):
    """Hybrid classical classifier using an RBF kernel and a feed‑forward net."""
    def __init__(self, num_features: int, depth: int, num_support: int = 20, gamma: float = 1.0) -> None:
        super().__init__()
        # The kernel maps inputs to a feature vector of length ``num_support``.
        self.kernel = Kernel(gamma)
        self.support_vectors = nn.Parameter(torch.randn(num_support, num_features), requires_grad=False)
        # The classifier consumes the kernel feature vector.
        self.classifier, _, _, _ = build_classifier_circuit(num_support, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for ``x``."""
        # Compute kernel features: (batch, num_support)
        k_features = torch.stack([self.kernel(x, sv) for sv in self.support_vectors], dim=1)
        return self.classifier(k_features)

    def fit(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-3, epochs: int = 100) -> None:
        """Simple training loop over the classical classifier."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            logits = self.forward(X)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

__all__ = ["SharedClassName", "kernel_matrix", "build_classifier_circuit"]
