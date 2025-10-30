"""Quantum kernel implementation using Pennylane.

This module mirrors the classical `QuantumKernelMethod` interface but
evaluates the kernel via a parameterised quantum circuit.  The circuit
is fixed by default but can be made learnable by passing
``learnable=True``.  The kernel value is the absolute value of the
overlap between the two encoded states.

The implementation is intentionally lightweight so that it can be
plugged into classical pipelines while still exposing quantum
capabilities.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml
import torch
from torch import nn


class QuantumKernelMethod(nn.Module):
    """
    Quantum RBF‑style kernel implemented with Pennylane.

    Parameters
    ----------
    n_qubits : int, default=4
        Number of qubits in the device.
    n_layers : int, default=2
        Number of repeat layers in the ansatz.
    device : str, default="default.qubit"
        Pennylane device name.
    backend : str, default="default.qubit"
        Pennylane backend used for statevector extraction.
    learnable : bool, default=False
        If ``True`` the rotation angles are optimised during training.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        device: str = "default.qubit",
        backend: str = "default.qubit",
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device, wires=n_qubits, shots=None)
        self.backend = backend
        self.learnable = learnable

        # Parameters for the rotation layers
        if self.learnable:
            self.params = nn.Parameter(
                torch.randn(n_layers, n_qubits, 3, dtype=torch.float32)
            )
        else:
            self.register_buffer("params", torch.randn(n_layers, n_qubits, 3))

        # Fixed entangling gates
        self.entanglers = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

    def _ansatz(self, vec: torch.Tensor) -> None:
        """
        Encode a classical vector `vec` into the quantum state.
        The encoding uses Ry rotations followed by a fixed entangling pattern.
        """
        for i in range(self.n_qubits):
            qml.RY(vec[i], wires=i)

        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RX(self.params[layer, i, 0], wires=i)
                qml.RY(self.params[layer, i, 1], wires=i)
                qml.RZ(self.params[layer, i, 2], wires=i)
            for a, b in self.entanglers:
                qml.CNOT(wires=[a, b])

    def _kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the absolute overlap between the states |x⟩ and |y⟩.
        """

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x_vec: torch.Tensor, y_vec: torch.Tensor) -> torch.Tensor:
            self._ansatz(x_vec)
            qml.adjoint(self._ansatz)(y_vec)
            return qml.expval(qml.PauliZ(0))

        # The circuit returns an expectation value in [-1,1]; map to [0,1]
        val = circuit(x, y)
        return torch.abs(val)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Public API matching the classical kernel.
        """
        return self._kernel_value(x, y)

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of vectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors.

        Returns
        -------
        np.ndarray
            2‑D array of shape (len(a), len(b)).
        """
        mat = torch.empty((len(a), len(b)), dtype=torch.float32)
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                mat[i, j] = self.forward(xi, yj)
        return mat.cpu().numpy()


def kernel_matrix(
    a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
) -> np.ndarray:
    """
    Compatibility shim for the legacy function.
    """
    kernel = QuantumKernelMethod(learnable=False)
    return kernel.kernel_matrix(a, b)


__all__ = ["QuantumKernelMethod", "kernel_matrix"]
