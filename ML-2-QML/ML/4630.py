"""QuantumKernelMethod - Classical hybrid interface.

This module provides a unified :class:`QuantumKernelMethod` that can operate in
three modes:

* ``kernel`` – a classical RBF kernel or a TorchQuantum kernel.
* ``qnn``   – a lightweight feed‑forward network mirroring Qiskit’s
  :class:`EstimatorQNN`.
* ``qcnn``  – a convolution‑inspired stack of linear layers mirroring the QCNN
  helper.

The public API is intentionally the same across the three modes so that the
quantum and classical implementations can be swapped without changing the
experiment code.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from typing import Sequence

__all__ = ["QuantumKernelMethod"]


# --------------------------------------------------------------------------- #
# Classical kernels and neural nets
# --------------------------------------------------------------------------- #
class ClassicalRBFKernel:
    """Fast NumPy implementation of the RBF kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the Gram matrix G_{ij} = exp(-γ‖a_i−b_j‖²)."""
        aa = np.sum(a**2, axis=1).reshape(-1, 1)
        bb = np.sum(b**2, axis=1).reshape(1, -1)
        sq_dist = aa - 2 * a @ b.T + bb
        return np.exp(-self.gamma * sq_dist)


class ClassicalEstimatorQNN(nn.Module):
    """Tiny feed‑forward regressor mimicking Qiskit’s EstimatorQNN."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class ClassicalQCNNModel(nn.Module):
    """Convolution‑style network that emulates the QCNN helper."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


# --------------------------------------------------------------------------- #
# Unified hybrid class
# --------------------------------------------------------------------------- #
class QuantumKernelMethod:
    """Hybrid interface that delegates to classical or quantum backends.

    Parameters
    ----------
    mode: {"kernel", "qnn", "qcnn"}
        Which backend to use.  ``kernel`` uses a pre‑computed kernel with an
        SVR, ``qnn`` uses a small MLP regressor, and ``qcnn`` uses an MLP
        classifier that mimics the QCNN layout.
    gamma: float, optional
        RBF kernel width (only used in ``kernel`` mode).
    """

    def __init__(self, mode: str = "kernel", gamma: float = 1.0) -> None:
        self.mode = mode
        if mode == "kernel":
            self.kernel = ClassicalRBFKernel(gamma)
            self.model = None
        elif mode == "qnn":
            self.model = ClassicalEstimatorQNN()
            self.scaler = StandardScaler()
        elif mode == "qcnn":
            self.model = ClassicalQCNNModel()
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported mode {mode}")

    # --------------------------------------------------------------------- #
    # Kernel utilities
    # --------------------------------------------------------------------- #
    def compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return the Gram matrix for ``X`` vs. ``Y``."""
        if self.mode!= "kernel":
            raise RuntimeError("Kernel matrix only available in 'kernel' mode")
        return self.kernel(X, Y)

    # --------------------------------------------------------------------- #
    # Training & prediction
    # --------------------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.mode == "kernel":
            # Train a support‑vector regressor with a pre‑computed kernel
            K = self.kernel(X, X)
            self.svr = SVR(kernel="precomputed")
            self.svr.fit(K, y)
        elif self.mode == "qnn":
            X_scaled = self.scaler.fit_transform(X)
            self.reg = MLPRegressor(hidden_layer_sizes=(8, 4),
                                    activation="tanh",
                                    max_iter=2000,
                                    random_state=42)
            self.reg.fit(X_scaled, y)
        elif self.mode == "qcnn":
            X_scaled = self.scaler.fit_transform(X)
            self.reg = MLPClassifier(hidden_layer_sizes=(16, 8, 4),
                                    activation="tanh",
                                    max_iter=2000,
                                    random_state=42)
            self.reg.fit(X_scaled, y)
        else:
            raise RuntimeError("Unsupported mode")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "kernel":
            K = self.kernel(X, self.svr.support_)
            return self.svr.predict(K)
        elif self.mode == "qnn":
            X_scaled = self.scaler.transform(X)
            return self.reg.predict(X_scaled)
        elif self.mode == "qcnn":
            X_scaled = self.scaler.transform(X)
            return self.reg.predict(X_scaled)
        else:
            raise RuntimeError("Unsupported mode")
