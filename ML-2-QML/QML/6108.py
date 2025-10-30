"""Quantum kernel construction using a parameterized variational circuit."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, RegressorMixin


class KernalAnsatz(tq.QuantumModule):
    """Parameterized ansatz that encodes two datasets and measures their overlap."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.ansatz_params = nn.Parameter(torch.randn(n_wires, dtype=torch.float32))
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # Encode x
        q_device.reset_states(x.shape[0])
        for i in range(self.n_wires):
            params = x[:, i] if x.shape[1] > i else None
            func_name_dict["ry"](q_device, wires=[i], params=params)
        # Entangle
        for i in range(self.n_wires - 1):
            func_name_dict["cx"](q_device, wires=[i, i + 1])
        # Encode y with negative parameters
        for i in range(self.n_wires):
            params = -y[:, i] if y.shape[1] > i else None
            func_name_dict["ry"](q_device, wires=[i], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(n_wires=n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.ansatz.q_device, x, y)
        return torch.abs(self.ansatz.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two batches of data using the quantum kernel."""
    kernel = Kernel()
    return np.array([[kernel(torch.tensor(x), torch.tensor(y)).item() for y in b] for x in a])


class QuantumKernelMethod(BaseEstimator, RegressorMixin):
    """Hybrid estimator that uses a variational quantum kernel and an SVM for regression."""
    def __init__(self, n_wires: int = 4, C: float = 1.0, lr: float = 0.01, epochs: int = 100):
        self.n_wires = n_wires
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self._svm: SVC | None = None
        self._kernel = Kernel(n_wires=self.n_wires)

    def _kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return kernel_matrix([torch.tensor(x) for x in X], [torch.tensor(y) for y in Y])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelMethod":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        K = self._kernel_matrix(X, X)
        self._svm = SVC(kernel="precomputed", C=self.C)
        self._svm.fit(K, y)
        self.X_train_ = X
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._svm is None:
            raise RuntimeError("The model must be fitted before calling predict.")
        X = np.asarray(X, dtype=np.float32)
        K = self._kernel_matrix(X, self.X_train_)
        return self._svm.predict(K)

    def train_params(self, X: np.ndarray, y: np.ndarray, target_kernel: np.ndarray | None = None) -> None:
        """Gradient‑based optimisation of the ansatz parameters."""
        if target_kernel is None:
            target_kernel = kernel_matrix([torch.tensor(x) for x in X], [torch.tensor(x) for x in X])
        optimizer = torch.optim.Adam(self._kernel.ansatz.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            K = self._kernel_matrix(X, X)
            loss = torch.mean((torch.tensor(K) - torch.tensor(target_kernel)) ** 2)
            loss.backward()
            optimizer.step()

    @property
    def params_(self) -> torch.Tensor:
        """Return the trained ansatz parameters."""
        return self._kernel.ansatz.ansatz_params.detach()


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "QuantumKernelMethod"]
