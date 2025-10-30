"""Variational quantum kernel using PennyLane."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml
import torch
from torch import nn


class QuantumKernel(nn.Module):
    """Quantum kernel based on a parameter‑shiftable variational ansatz.

    The ansatz encodes each datum into a quantum state on ``n_wires`` qubits and the kernel
    is evaluated as the squared fidelity between the two states.

    Parameters
    ----------
    n_wires : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    backend : str, optional
        PennyLane simulator to use (default: ``'default.qubit'``).
    """

    def __init__(
        self,
        n_wires: int = 4,
        n_layers: int = 2,
        backend: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.device = qml.device(backend, wires=n_wires)

        # Trainable parameters for the ansatz
        self.params = nn.Parameter(
            torch.randn(n_layers, n_wires, 3)  # 3 rotation angles per qubit
        )

        def _circuit(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Data‑encoding layer
            for i in range(n_wires):
                qml.RY(x[i], wires=i)
            # Variational layers
            for layer in range(n_layers):
                for w in range(n_wires):
                    qml.RX(params[layer, w, 0], wires=w)
                    qml.RY(params[layer, w, 1], wires=w)
                    qml.RZ(params[layer, w, 2], wires=w)
                # Entangling layer
                for w in range(n_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
                qml.CNOT(wires=[n_wires - 1, 0])  # wrap‑around
            return qml.state()

        self.circuit = qml.QNode(_circuit, self.device, interface="torch")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the quantum kernel value k(x, y) = |⟨ψ(x)|ψ(y)⟩|²."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        psi_x = self.circuit(x[0], self.params)
        psi_y = self.circuit(y[0], self.params)
        # Overlap
        overlap = torch.abs(torch.dot(psi_x.conj(), psi_y)) ** 2
        return overlap

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix for two batches of tensors."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["QuantumKernel", "kernel_matrix"]
