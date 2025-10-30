"""ConvGen025: Classical implementation of a hybrid conv‑reg‑kernel module.

The class combines a 2‑D convolutional filter, a multi‑layer perceptron,
an RBF kernel, and a binary classifier.  It is a drop‑in replacement
for the original Conv.py while exposing a unified API that mirrors
the quantum interface.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple, Sequence

# Classical convolution filter
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        tensor = data.reshape(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

# Classical regression model
class MLModel(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# Classical RBF kernel
class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    k = Kernel(gamma)
    return np.array([[k(x, y).item() for y in b] for x in a])

# Classical classifier
class Classifier(nn.Module):
    def __init__(self, num_features: int, depth: int = 2) -> None:
        super().__init__()
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_features, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Unified class
class ConvGen025:
    """Unified classical interface mirroring the quantum counterpart."""
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 num_features: int = 10,
                 gamma: float = 1.0,
                 classifier_depth: int = 2):
        self.filter = ConvFilter(kernel_size, threshold)
        self.regressor = MLModel(num_features)
        self.kernel = Kernel(gamma)
        self.classifier = Classifier(num_features, classifier_depth)

    def filter_data(self, data: np.ndarray) -> float:
        tensor = torch.tensor(data, dtype=torch.float32)
        return self.filter(tensor).item()

    def regress(self, x: np.ndarray) -> float:
        tensor = torch.tensor(x, dtype=torch.float32)
        return self.regressor(tensor).item()

    def compute_kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_t = torch.tensor(a, dtype=torch.float32)
        b_t = torch.tensor(b, dtype=torch.float32)
        return kernel_matrix(a_t, b_t, self.kernel.gamma)

    def classify(self, x: np.ndarray) -> torch.Tensor:
        tensor = torch.tensor(x, dtype=torch.float32)
        return self.classifier(tensor)

__all__ = ["ConvGen025"]
