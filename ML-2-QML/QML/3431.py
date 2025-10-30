"""Hybrid kernel and estimator using TorchQuantum.

The class implements the same API as the classical counterpart but evaluates the
quantum kernel via a variational circuit and provides a QNN estimator.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict, pauli

# -------------------------------------------------------------
# Variational ansatz used for the quantum kernel
# -------------------------------------------------------------
class VariationalAnsatz(tq.QuantumModule):
    """Programmable variational circuit with Ry and CX layers."""
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.func_list = []
        for d in range(depth):
            for i in range(n_wires):
                self.func_list.append({"input_idx": [i], "func": "ry", "wires": [i]})
            for i in range(n_wires - 1):
                self.func_list.append({"input_idx": [], "func": "cx", "wires": [i, i+1]})

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if info["input_idx"] else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if info["input_idx"] else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

# -------------------------------------------------------------
# Quantum kernel
# -------------------------------------------------------------
class QuantumKernel(tq.QuantumModule):
    """Quantum kernel with variational ansatz."""
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = VariationalAnsatz(n_wires, depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

# -------------------------------------------------------------
# EstimatorQNN
# -------------------------------------------------------------
class EstimatorQNN(tq.QuantumModule):
    """Variational quantum neural network for regression."""
    def __init__(self, n_wires: int = 1) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.weight = torch.nn.Parameter(torch.randn(1, 1))
        self.observable = pauli('Y', [0])

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.q_device.reset_states(x.shape[0])
        self.q_device.ry(x[:, 0], 0, params=self.weight)
        return self.q_device.expectation(self.observable)

# -------------------------------------------------------------
# Combined hybrid estimator
# -------------------------------------------------------------
class HybridQuantumKernelEstimator(tq.QuantumModule):
    """Hybrid kernel + QNN with the same API as the classical counterpart."""
    def __init__(self, n_wires: int = 4, depth: int = 2,
                 gamma: float = 1.0, quantum_weight: float = 0.5):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.gamma = gamma
        self.quantum_weight = quantum_weight
        self.kernel = QuantumKernel(n_wires, depth)
        self.estimator = EstimatorQNN(n_wires)

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def _quantum(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        rbf_val = self._rbf(x, y)
        q_val = self._quantum(x, y)
        return self.quantum_weight * q_val + (1.0 - self.quantum_weight) * rbf_val

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def fit(self, X: torch.Tensor, y: torch.Tensor, alpha: float = 1e-3) -> None:
        K = self.kernel_matrix(X, X)
        self.coef_ = torch.linalg.solve(torch.tensor(K) + alpha * torch.eye(len(X)),
                                        y.unsqueeze(-1)).squeeze()

    def predict(self, X: torch.Tensor, X_train: torch.Tensor) -> torch.Tensor:
        K_test = self.kernel_matrix(X, X_train)
        return K_test @ self.coef_

__all__ = ["HybridQuantumKernelEstimator", "EstimatorQNN"]
