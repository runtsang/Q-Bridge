import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

class RBFKernel(nn.Module):
    """Classical radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class QCNNModel(nn.Module):
    """Fully‑connected surrogate for a QCNN."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridKernelQCNN:
    """
    Hybrid model that first maps data through a classical RBF kernel
    and then feeds the resulting kernel vector into a QCNN‑style network.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        self.kernel = RBFKernel(gamma)
        self.qcnn = QCNNModel()
        self._support: Sequence[torch.Tensor] | None = None

    def fit(self, X: torch.Tensor) -> None:
        """
        Store support vectors for kernel evaluation.
        """
        self._support = [x for x in X]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._support is None:
            raise RuntimeError("HybridKernelQCNN must be fitted before calling forward.")
        # Compute kernel vector between support set and new sample
        k_mat = kernel_matrix(self._support, [x], gamma=self.kernel.gamma)
        k_vec = torch.tensor(k_mat.squeeze(), dtype=torch.float32)
        # Pass through QCNN surrogate
        return self.qcnn(k_vec)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix using an RBF kernel."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernelQCNN", "kernel_matrix"]
