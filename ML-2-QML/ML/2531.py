"""Classical and hybrid kernel classifier.

The module exposes a single :class:`KernelClassifier` that can operate
in two modes:

* ``use_quantum=False`` – classical RBF kernel + scikit‑learn SVC.
* ``use_quantum=True``  – quantum kernel built with TorchQuantum
  (Ry‑encoding) + scikit‑learn SVC.

The API is identical in both cases, so downstream code can swap the
implementation without modification.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------
# Classical RBF kernel
# ----------------------------------------------------------------------
class _RBFKernel(nn.Module):
    """Pure‑Python RBF kernel implemented in PyTorch for fast
    batched evaluation."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (n, d), y: (m, d)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n, m, d)
        sq_dist = torch.sum(diff * diff, dim=2)  # (n, m)
        return torch.exp(-self.gamma * sq_dist)


# ----------------------------------------------------------------------
# Quantum kernel (TorchQuantum)
# ----------------------------------------------------------------------
class _QuantumKernel(nn.Module):
    """Quantum kernel built from a Ry‑encoding ansatz.

    The kernel value is |⟨0|U(x)†U(y)|0⟩|², where U(x) is the
    parameterised circuit that encodes the feature vector ``x``.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        import torchquantum as tq
        self.tq = tq

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (n, d), y: (m, d)
        n, d = x.shape
        m, _ = y.shape
        device = self.tq.QuantumDevice(n_wires=self.n_wires)
        kernel = torch.empty(n, m)
        for i in range(n):
            xi = x[i]
            for j in range(m):
                yj = y[j]
                device.reset_states(1)
                # encode x
                for k, val in enumerate(xi):
                    device.apply(self.tq.RY, wires=k, params=val)
                # encode -y
                for k, val in enumerate(yj):
                    device.apply(self.tq.RY, wires=k, params=-val)
                kernel[i, j] = torch.abs(device.states[0]).item()
        return kernel


# ----------------------------------------------------------------------
# Unified classifier
# ----------------------------------------------------------------------
class KernelClassifier:
    """Hybrid kernel‑based classifier.

    Parameters
    ----------
    gamma : float, optional
        RBF kernel width. Ignored when ``use_quantum=True``.
    depth : int, optional
        Depth of the quantum circuit (only used when ``use_quantum=True``).
    use_quantum : bool, default=False
        If ``True`` the quantum kernel is used; otherwise a classical
        RBF kernel is employed.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        depth: int = 2,
        use_quantum: bool = False,
    ) -> None:
        self.gamma = gamma
        self.depth = depth
        self.use_quantum = use_quantum
        self.scaler = StandardScaler()
        self.svm = SVC(kernel="precomputed")
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        # kernel modules
        self._rbf_kernel = _RBFKernel(gamma=self.gamma)
        self._quantum_kernel: Optional[_QuantumKernel] = None

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def _kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return the Gram matrix between X and Y."""
        if self.use_quantum:
            if self._quantum_kernel is None:
                self._quantum_kernel = _QuantumKernel(n_wires=X.shape[1])
            X_t = torch.from_numpy(X).float()
            Y_t = torch.from_numpy(Y).float()
            return self._quantum_kernel(X_t, Y_t).numpy()
        else:
            X_t = torch.from_numpy(X).float()
            Y_t = torch.from_numpy(Y).float()
            return self._rbf_kernel(X_t, Y_t).numpy()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the SVM with a pre‑computed kernel matrix."""
        X = self.scaler.fit_transform(X)
        self.X_train = X
        self.y_train = y
        K = self._kernel_matrix(X, X)
        self.svm.fit(K, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data."""
        X = self.scaler.transform(X)
        K = self._kernel_matrix(X, self.X_train)
        return self.svm.predict(K)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the decision function values."""
        X = self.scaler.transform(X)
        K = self._kernel_matrix(X, self.X_train)
        return self.svm.decision_function(K)

__all__ = ["KernelClassifier"]
