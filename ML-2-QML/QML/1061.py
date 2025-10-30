"""Quantum kernel with tunable rotation schedule and entangling block."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence, List


class QuantumAnsatz(tq.QuantumModule):
    """Programmable ansatz with trainable rotation angles and an entangling gate."""
    def __init__(self, n_wires: int, params: List[float]):
        super().__init__()
        self.n_wires = n_wires
        self.params = nn.Parameter(torch.tensor(params, dtype=torch.float32))
        self.entangle = tq.CNOT  # simple entanglement

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode x with trainable rotations
        for i in range(self.n_wires):
            angle = self.params[i] * x[:, i]
            tq.ry(q_device, wires=i, params=angle)
        # Entangle
        self.entangle(q_device, wires=[0, 1])
        # Encode y with inverse rotations
        for i in range(self.n_wires):
            angle = -self.params[i] * y[:, i]
            tq.ry(q_device, wires=i, params=angle)


class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states."""
    def __init__(self, n_wires: int = 4, init_params: List[float] = None):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumAnsatz(n_wires, init_params or [np.pi/4]*self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 50, lr: float = 1e-3) -> None:
        """Train the rotation parameters to maximize kernel alignment with labels."""
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            preds = self.forward(X, X).diag()
            loss = torch.mean((preds - y)**2)
            loss.backward()
            opt.step()

    def predict(self, X: torch.Tensor, X_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        """Kernel ridge regression prediction using the learned quantum kernel."""
        K = self.forward(X_train, X_train)
        alpha = torch.linalg.solve(K + 1e-5 * torch.eye(K.shape[0]), y_train)
        K_test = self.forward(X, X_train)
        return K_test @ alpha


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = [
    "QuantumAnsatz",
    "QuantumKernel",
    "kernel_matrix",
]
