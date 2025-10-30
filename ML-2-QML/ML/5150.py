from __future__ import annotations

from typing import Iterable, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalKernel(nn.Module):
    """Classical RBF kernel with trainable gamma and optional feature map."""
    def __init__(self, gamma: float = 1.0, feature_map: nn.Module | None = None):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.feature_map = feature_map

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Map inputs through an optional feature map before computing RBF."""
        if self.feature_map:
            x = self.feature_map(x)
            y = self.feature_map(y)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class HybridHead(nn.Module):
    """Differentiable sigmoid head that mimics the quantum expectation."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)


class FeedForwardClassifier(nn.Module):
    """Deep feed‑forward classifier that shares metadata with the quantum circuit."""
    def __init__(self, num_features: int, depth: int):
        super().__init__()
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(num_features, num_features))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(num_features, num_features))
            layers.append(nn.ReLU())
            # store weight sizes for later inspection
            layers.append(nn.Linear(num_features, num_features))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.encoding = list(range(num_features))
        self.weight_sizes = [p.numel() for p in self.network.parameters()]
        self.observables = list(range(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[FeedForwardClassifier, Iterable[int], Iterable[int], list[int]]:
    """Return the classical counterpart to the quantum circuit factory."""
    classifier = FeedForwardClassifier(num_features, depth)
    encoding = classifier.encoding
    weight_sizes = classifier.weight_sizes
    return classifier, encoding, weight_sizes, classifier.observables


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute a Gram matrix using the classical RBF kernel."""
    kernel = ClassicalKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["ClassicalKernel", "HybridHead", "FeedForwardClassifier", "build_classifier_circuit", "kernel_matrix"]
