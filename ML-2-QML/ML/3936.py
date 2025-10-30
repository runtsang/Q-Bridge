"""Hybrid kernel and QCNN implementation that blends classical and quantum
components into a single API.

The module contains:
  * Classical RBF kernel utilities (``RBFKernel`` and ``RBFKernelModule``).
  * A convolution‑inspired fully‑connected network (``QCNNModel``).
  * A high‑level façade (``HybridKernelQCNN``) that exposes both the
    classical and quantum kernels, the QCNN forward passes, and a
    concatenated hybrid feature space.

The design is minimal yet fully importable and is intentionally
compatible with the original “QuantumKernelMethod.py” and
“QCNN.py” seeds while extending them with a unified interface.
"""

import numpy as np
import torch
from torch import nn
from typing import Sequence

# Classical RBF kernel -------------------------------------------------------

class RBFKernel(nn.Module):
    """Exponentiated quadratic kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class RBFKernelModule(nn.Module):
    """Ensures 2‑D tensors before delegating to ``RBFKernel``."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = RBFKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = RBFKernelModule(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# Classical QCNN -------------------------------------------------------------

class QCNNModel(nn.Module):
    """Fully‑connected network that mimics the QCNN layer pattern."""
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

def QCNN() -> QCNNModel:
    """Factory that returns a new ``QCNNModel`` instance."""
    return QCNNModel()

# Hybrid façade --------------------------------------------------------------

class HybridKernelQCNN:
    """
    Unified interface that exposes:
      * Classical RBF kernel matrix.
      * Quantum kernel matrix (via the qml module).
      * Classical QCNN forward pass.
      * Quantum QCNN forward pass (EstimatorQNN).
      * Hybrid feature concatenation.
    """
    def __init__(self, gamma: float = 1.0, n_wires: int = 4) -> None:
        self.gamma = gamma
        self.n_wires = n_wires
        # Classical parts
        self.kernel = RBFKernelModule(gamma)
        self.qcnn = QCNN()
        # Quantum parts – imported lazily to avoid heavy imports at module load
        from.QuantumKernelMethod__gen252_qml import QuantumHybridKernelQCNNQML
        self.qml = QuantumHybridKernelQCNNQML(gamma=gamma, n_wires=n_wires)

    def classical_features(self, X: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(X, X, gamma=self.gamma)

    def quantum_features(self, X: Sequence[torch.Tensor]) -> np.ndarray:
        return self.qml.kernel_matrix(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.qcnn(X)

    def predict_quantum(self, X: torch.Tensor) -> torch.Tensor:
        return self.qml.predict(X)

    def hybrid_features(self, X: Sequence[torch.Tensor]) -> np.ndarray:
        Kc = self.classical_features(X)
        Kq = self.quantum_features(X)
        return np.concatenate([Kc, Kq], axis=1)

__all__ = [
    "RBFKernel",
    "RBFKernelModule",
    "kernel_matrix",
    "QCNNModel",
    "QCNN",
    "HybridKernelQCNN",
]
