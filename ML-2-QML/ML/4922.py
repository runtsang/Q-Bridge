"""
Hybrid QCNN – classical implementation.
Provides a feature extractor that combines convolutional layers,
a classical RBF kernel and a regression head.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
#  Core kernel – classical RBF
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """RBF kernel implemented as a PyTorch module for easy composition."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that normalises inputs and exposes a simple forward API."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.ansatz(x.view(1, -1), y.view(1, -1)).squeeze()

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two lists of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
#  Simple regression dataset (mirroring QuantumRegression.py)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Torch dataset that generates superposition data on the fly."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
#  Hybrid QCNN – classical path
# --------------------------------------------------------------------------- #
class QCNNGen113(nn.Module):
    """
    Classical QCNN that emulates the quantum architecture using fully‑connected
    layers.  It augments the learned features with an RBF kernel computed
    between the batch and a set of prototype vectors, then feeds the result
    into a small regression head.
    """
    def __init__(
        self,
        input_dim: int = 8,
        gamma: float = 1.0,
        prototype_count: int = 10,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)

        # ------------------------------------------------------------------ #
        # Feature extraction – mimic QCNN’s convolution & pooling
        # ------------------------------------------------------------------ #
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # ------------------------------------------------------------------ #
        # Kernel module – RBF
        # ------------------------------------------------------------------ #
        self.kernel = Kernel(gamma)

        # Random prototypes used to evaluate the kernel matrix
        self.register_buffer(
            "prototypes",
            torch.randn(prototype_count, 16, dtype=torch.float32),
        )

        # ------------------------------------------------------------------ #
        # Regression head
        # ------------------------------------------------------------------ #
        self.head = nn.Linear(16 + prototype_count, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Compute kernel between batch and prototypes
        # Shape: (batch, prototypes)
        kernel_features = torch.cat(
            [
                self.kernel(x[i : i + 1], self.prototypes).squeeze(0)
                for i in range(x.shape[0])
            ],
            dim=0,
        )
        # Concatenate learned features with kernel statistics
        combined = torch.cat([x, kernel_features], dim=1)
        return torch.sigmoid(self.head(combined))

def QCNNGen113() -> QCNNGen113:
    """Factory returning a ready‑to‑train instance."""
    return QCNNGen113()

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "RegressionDataset", "generate_superposition_data",
           "QCNNGen113", "QCNNGen113"]
