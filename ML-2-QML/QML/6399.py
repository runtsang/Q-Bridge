"""Quantum kernel construction with a trainable variational ansatz and amplitude‑based similarity."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.svm import SVC

__all__ = ["QuantumAnsatz", "QuantumKernel", "kernel_matrix", "QuantumKernelClassifier"]

class QuantumAnsatz(tq.QuantumModule):
    """Variational circuit with trainable rotation angles."""
    def __init__(self, n_wires: int, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.params = nn.Parameter(torch.randn(depth * n_wires, 1))
        self.build_circuit()

    def build_circuit(self) -> None:
        self.circuit = []
        for d in range(self.depth):
            for w in range(self.n_wires):
                self.circuit.append({"func": "ry", "wires": [w], "param_idx": d * self.n_wires + w})

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for gate in self.circuit:
            idx = gate["param_idx"]
            param = self.params[idx] * x[:, idx % self.n_wires]
            func_name_dict[gate["func"]](q_device, wires=gate["wires"], params=param)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that evaluates the amplitude overlap between two encoded states."""
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumAnsatz(n_wires=self.n_wires, depth=depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x)
        amp_x = self.q_device.states.view(-1)[0]
        self.ansatz(self.q_device, y)
        amp_y = self.q_device.states.view(-1)[0]
        return torch.abs(amp_x * amp_y.conj()).squeeze()

def kernel_matrix(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
    """Compute the Gram matrix using the quantum kernel."""
    a = np.vstack(a)
    b = np.vstack(b)
    kernel = QuantumKernel()
    return np.array([[kernel(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)).item() for y in b] for x in a])

class QuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    """Classifier that trains the variational ansatz and uses an SVC with pre‑computed kernel."""
    def __init__(self, lr: float = 0.01, epochs: int = 200):
        self.lr = lr
        self.epochs = epochs
        self.kernel_ = None
        self.svc_ = None
        self.X_ = None

    def fit(self, X: Sequence[np.ndarray], y: Sequence[int]) -> "QuantumKernelClassifier":
        X, y = check_X_y(X, y)
        self.X_ = X
        self.kernel_ = QuantumKernel()
        opt = torch.optim.Adam(self.kernel_.parameters(), lr=self.lr)

        # Simple contrastive loss: maximize kernel for same class, minimize for different
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = 0.0
            for i, xi in enumerate(X):
                for j, xj in enumerate(X):
                    k = self.kernel_(torch.tensor(xi, dtype=torch.float32),
                                    torch.tensor(xj, dtype=torch.float32))
                    if y[i] == y[j]:
                        loss -= k
                    else:
                        loss += k
            loss.backward()
            opt.step()

        gram = kernel_matrix(X, X)
        self.svc_ = SVC(kernel="precomputed")
        self.svc_.fit(gram, y)
        return self

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        if self.svc_ is None or self.X_ is None:
            raise ValueError("The model has not been fitted yet.")
        gram = kernel_matrix(X, self.X_)
        return self.svc_.predict(gram)
