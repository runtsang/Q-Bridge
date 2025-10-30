"""Quantum kernel module using PennyLane."""

from __future__ import annotations

from typing import Sequence, Optional

import pennylane as qml
import torch

__all__ = ["QuantumKernelMethod", "kernel_matrix"]


class QuantumKernelMethod:
    """
    Kernel that can evaluate a pure quantum kernel, a classical RBF kernel,
    or a weighted hybrid of the two.
    """
    def __init__(
        self,
        mode: str = "quantum",
        gamma: Optional[float] = None,
        alpha: float = 0.5,
        n_wires: int = 4,
        circuit: Optional[qml.QNode] = None,
    ):
        if mode not in {"classical", "quantum", "hybrid"}:
            raise ValueError(f"Unsupported mode {mode!r}")
        self.mode = mode
        self.alpha = alpha
        self.n_wires = n_wires
        self.gamma = gamma

        self.dev = qml.device("default.qubit", wires=self.n_wires)

        if circuit is None:
            @qml.qnode(self.dev, interface="torch")
            def default_circuit(x):
                for i in range(self.n_wires):
                    qml.RY(x[i], wires=i)
                for i in range(self.n_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                return qml.state()
            self.circuit = default_circuit
        else:
            self.circuit = circuit

    def _quantum_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        state_x = self.circuit(x)
        state_y = self.circuit(y)
        overlap = torch.abs(torch.vdot(state_x, state_y)) ** 2
        return overlap

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        sq_dist = torch.sum(diff * diff, dim=-1, keepdim=True)
        gamma = self.gamma if self.gamma is not None else 1.0
        return torch.exp(-gamma * sq_dist)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "classical":
            return self._rbf_kernel(x, y).squeeze()
        if self.mode == "quantum":
            return self._quantum_kernel(x, y).squeeze()
        rbf = self._rbf_kernel(x, y).squeeze()
        q = self._quantum_kernel(x, y).squeeze()
        return self.alpha * rbf + (1 - self.alpha) * q

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        a = torch.stack(a)
        b = torch.stack(b)
        mat = torch.zeros((len(a), len(b)), dtype=torch.float32)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = self.forward(x, y)
        return mat.numpy()
