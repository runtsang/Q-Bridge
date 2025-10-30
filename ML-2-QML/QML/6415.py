"""Quantum kernel using PennyLane with a parameter‑tunable ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import pennylane as qml


class QuantumKernelMethod(nn.Module):
    """Quantum kernel based on a simple data‑encoding circuit.

    The kernel value is the probability of measuring the all‑zero
    computational basis state after encoding two feature vectors
    and applying the inverse of the second encoding.  This is a
    fidelity‑based kernel that captures quantum similarity.
    """

    def __init__(self, n_qubits: int = 4, depth: int = 1, device_name: str = "default.qubit") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.device = qml.device(device_name, wires=n_qubits)

        @qml.qnode(self.device, interface="torch")
        def _kernel(params_x: torch.Tensor, params_y: torch.Tensor) -> torch.Tensor:
            # Encode the first vector
            for i, w in enumerate(params_x):
                qml.RY(w, wires=i)
            # Entangling layer
            for d in range(self.depth):
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Apply the inverse of the second vector
            for i, w in enumerate(params_y):
                qml.RY(-w, wires=i)
            # Probability of all zeros
            return qml.probs(wires=range(self.n_qubits))[0]

        self._kernel = _kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value for a single pair of vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of length >= ``n_qubits`` (only the first
            ``n_qubits`` elements are used for encoding).

        Returns
        -------
        torch.Tensor
            Kernel value in the interval [0, 1].
        """
        x = x[: self.n_qubits]
        y = y[: self.n_qubits]
        return self._kernel(x, y)

    @staticmethod
    def kernel_matrix(
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        n_qubits: int = 4,
        depth: int = 1,
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of feature vectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Each element should be a 1‑D tensor of length >= ``n_qubits``.
        n_qubits : int, optional
            Number of qubits used for encoding.
        depth : int, optional
            Depth of the entangling layer.

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(device, interface="torch")
        def _kernel(params_x: torch.Tensor, params_y: torch.Tensor) -> torch.Tensor:
            for i, w in enumerate(params_x):
                qml.RY(w, wires=i)
            for d in range(depth):
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            for i, w in enumerate(params_y):
                qml.RY(-w, wires=i)
            return qml.probs(wires=range(n_qubits))[0]

        K = np.zeros((len(a), len(b)))
        for i, ax in enumerate(a):
            for j, by in enumerate(b):
                K[i, j] = _kernel(ax[:n_qubits], by[:n_qubits]).item()
        return K


__all__ = ["QuantumKernelMethod"]
