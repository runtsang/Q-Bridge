"""Quantum kernel using a parameter‑shift ansatz and Pennylane."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import pennylane as qml

class KernalAnsatz:
    """Parameterized feature map and variational ansatz."""
    def __init__(self, n_wires: int = 4, depth: int = 1) -> None:
        self.n_wires = n_wires
        self.depth = depth
        # trainable parameters for the variational circuit
        param_shape = (depth, n_wires, 3)  # 3 rotation angles per qubit per layer
        self.params = torch.nn.Parameter(torch.randn(param_shape, dtype=torch.float32))
        # Pennylane device
        self.dev = qml.device("default.qubit", wires=n_wires)

    def circuit(self, data: np.ndarray, params: np.ndarray) -> None:
        """Encode data and apply variational layers."""
        # data encoding via Ry gates
        for w in range(self.n_wires):
            qml.RY(data[w], wires=w)
        # variational layers
        for d in range(self.depth):
            for w in range(self.n_wires):
                qml.RX(params[d, w, 0], wires=w)
                qml.RY(params[d, w, 1], wires=w)
                qml.RZ(params[d, w, 2], wires=w)
            # entangling layer
            for w in range(self.n_wires - 1):
                qml.CNOT(wires=[w, w + 1])

    @qml.qnode
    def qnode(self, data: np.ndarray, params: np.ndarray) -> np.ndarray:
        self.circuit(data, params)
        return qml.state()

class Kernel:
    """Quantum kernel that computes the absolute value of the overlap of two encoded states."""
    def __init__(self, n_wires: int = 4, depth: int = 1) -> None:
        self.n_wires = n_wires
        self.depth = depth
        self.ansatz = KernalAnsatz(n_wires, depth)
        # expose the parameters as a torch.nn.Parameter for differentiability
        self.params = torch.nn.Parameter(self.ansatz.params.clone())

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x if x.ndim == 2 else x.unsqueeze(0)
        y = y if y.ndim == 2 else y.unsqueeze(0)
        batch_x = x.shape[0]
        batch_y = y.shape[0]
        K = torch.zeros((batch_x, batch_y), dtype=torch.float32)
        for i in range(batch_x):
            for j in range(batch_y):
                state_x_np = self.ansatz.qnode(x[i].numpy(), self.params.detach().numpy())
                state_y_np = self.ansatz.qnode(y[j].numpy(), self.params.detach().numpy())
                state_x = torch.tensor(state_x_np, dtype=torch.complex64)
                state_y = torch.tensor(state_y_np, dtype=torch.complex64)
                overlap = torch.abs(torch.vdot(state_x, state_y)) ** 2
                K[i, j] = overlap
        return K

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_wires: int = 4, depth: int = 1) -> np.ndarray:
    """Compute Gram matrix between two lists of tensors using the quantum kernel."""
    kernel = Kernel(n_wires, depth)
    A = torch.stack(a).float()
    B = torch.stack(b).float()
    K = kernel.forward(A, B).detach().numpy()
    return K

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
