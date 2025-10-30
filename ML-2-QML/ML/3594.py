"""Hybrid classical kernel module integrating QCNN feature extraction and RBF similarity.

Provides a reusable :class:`HybridKernelModel` that can be used in kernel-based learning
algorithms. The model first projects inputs through a lightweight QCNN-inspired
fully‑connected network, then evaluates a Gaussian RBF kernel on the learned
representations.  This mirrors the quantum kernel implementation below and
enables side‑by‑side comparisons of classical vs quantum feature maps.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

class QCNNModel(nn.Module):
    """Classical surrogate of the quantum convolution‑pooling network.

    The architecture mimics the depth and layer sizes of the QCNN ansatz,
    providing a deterministic, differentiable map from raw data to a
    4‑dimensional feature vector used by the hybrid kernel.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x

class HybridKernelModel(nn.Module):
    """Hybrid RBF kernel built on top of a QCNN feature extractor.

    Parameters
    ----------
    gamma : float, optional
        Width parameter of the Gaussian kernel.  Default ``1.0``.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.feature_extractor = QCNNModel()

    def _feature(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw input to a 4‑dimensional feature vector."""
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the RBF kernel value between two examples."""
        fx = self._feature(x.reshape(1, -1))
        fy = self._feature(y.reshape(1, -1))
        diff = fx - fy
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: np.ndarray, b: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix for two collections of raw samples.

    Parameters
    ----------
    a, b : np.ndarray
        2‑D arrays of shape (n_samples, n_features).
    gamma : float, optional
        Width of the Gaussian kernel.
    """
    model = HybridKernelModel(gamma)
    a_t = torch.from_numpy(a.astype(np.float32))
    b_t = torch.from_numpy(b.astype(np.float32))
    return np.array([[model(x, y).item() for y in b_t] for x in a_t])

__all__ = ["QCNNModel", "HybridKernelModel", "kernel_matrix"]
