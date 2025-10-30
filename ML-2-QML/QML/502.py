"""Quantum kernel implementation using PennyLane.

This module builds a variational quantum kernel that can be tuned by the
number of qubits, layers, and the choice of single‑qubit rotation.
It exposes the same public API as the classical counterpart for easy
comparison and hybrid experiments.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
import torch
from typing import Sequence, Union

class QuantumKernelMethod:
    """Variational quantum kernel.

    Parameters
    ----------
    n_qubits : int, default=4
        Number of qubits used in the circuit.
    layers : int, default=2
        Number of variational layers applied after encoding.
    ansatz : str, default='ry'
        Single‑qubit rotation type used in the encoding and variational layers.
        Supported values: ``'ry'`` and ``'rz'``.
    dev : pennylane.Device, optional
        Quantum device. If None, a default ``default.qubit`` simulator is used.

    Notes
    -----
    The kernel value is the squared absolute overlap between the quantum
    states prepared from two input feature vectors. The same variational
    parameters are used for all evaluations, providing a fixed embedding
    space.
    """

    def __init__(self,
                 n_qubits: int = 4,
                 layers: int = 2,
                 ansatz: str = "ry",
                 dev: qml.Device | None = None) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        if ansatz.lower() not in {"ry", "rz"}:
            raise ValueError(f"Unsupported ansatz: {ansatz}")
        self.ansatz = ansatz.lower()
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)
        self._init_params()
        self._build_circuit()

    def _init_params(self) -> None:
        # Fixed variational parameters shared across all evaluations
        self.params = torch.randn(self.n_qubits, dtype=torch.float64, requires_grad=False)

    def _build_circuit(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Encode first vector
            for i in range(self.n_qubits):
                if self.ansatz == "ry":
                    qml.RY(x[i], wires=i)
                else:  # rz
                    qml.RZ(x[i], wires=i)
            # Variational layers
            for _ in range(self.layers):
                for i in range(self.n_qubits):
                    qml.RY(self.params[i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Un‑encode second vector (reverse operation)
            for i in range(self.n_qubits):
                if self.ansatz == "ry":
                    qml.RY(-y[i], wires=i)
                else:
                    qml.RZ(-y[i], wires=i)
            return qml.state()

        self._circuit = circuit

    def __call__(self,
                 x: Union[np.ndarray, torch.Tensor],
                 y: Union[np.ndarray, torch.Tensor]) -> float:
        """Return the kernel value for two feature vectors."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64)

        # Prepare state for x and y separately
        state_x = self._circuit(x, torch.zeros_like(x))
        state_y = self._circuit(torch.zeros_like(y), y)

        # Overlap magnitude squared
        overlap = torch.abs(torch.dot(state_x, state_y.conj())) ** 2
        return float(overlap.item())

    def kernel_matrix(self,
                      a: Sequence[Union[np.ndarray, torch.Tensor]],
                      b: Sequence[Union[np.ndarray, torch.Tensor]]) -> np.ndarray:
        """Compute the Gram matrix between two collections of samples."""
        a_np = np.asarray(a, dtype=float)
        b_np = np.asarray(b, dtype=float)
        mat = np.empty((len(a_np), len(b_np)), dtype=float)
        for i, x in enumerate(a_np):
            for j, y in enumerate(b_np):
                mat[i, j] = self(x, y)
        return mat

__all__ = ["QuantumKernelMethod"]
