"""Quantum kernel module using Pennylane variational circuit."""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from torch import nn
from typing import Sequence

class QuantumKernelMethod:
    """
    Quantum kernel implemented with a parameterized variational circuit and swap test.

    The circuit encodes two input vectors x and y in separate registers and measures the
    overlap of the resulting quantum states via a swap test. The parameters of the ansatz
    are trainable and optimized via gradient descent.

    Attributes
    ----------
    n_wires : int
        Number of qubits used for each input register.
    dev : pennylane.Device
        PennyLane quantum device (default: 'default.qubit').
    params : torch.nn.Parameter
        Trainable parameters of the variational ansatz.
    """

    def __init__(self, n_wires: int = 4, dev_name: str = "default.qubit") -> None:
        self.n_wires = n_wires
        self.dev = qml.device(dev_name, wires=2 * n_wires + 1)  # ancilla + two registers
        # initialise trainable parameters for a simple entangling layer
        self.params = nn.Parameter(torch.randn(n_wires, 2))
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, diff_method="backprop")
        def kernel_circuit(x: np.ndarray, y: np.ndarray):
            # ancilla qubit
            qml.Hadamard(wires=0)
            # encode x on first register
            for i in range(self.n_wires):
                qml.RY(x[i], wires=i + 1)
            # encode y on second register
            for i in range(self.n_wires):
                qml.RY(y[i], wires=i + 1 + self.n_wires)
            # entangle with trainable rotations
            for i in range(self.n_wires):
                qml.CNOT(wires=[i + 1, i + 1 + self.n_wires])
                qml.RZ(self.params[i, 0], wires=i + 1)
                qml.RZ(self.params[i, 1], wires=i + 1 + self.n_wires)
            # swap test
            for i in range(self.n_wires):
                qml.CSWAP(wires=[0, i + 1, i + 1 + self.n_wires])
            qml.Hadamard(wires=0)
            return (qml.expval(qml.PauliZ(0)) + 1) / 2
        self.kernel_circuit = kernel_circuit

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel between two input vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of shape (n_wires,).

        Returns
        -------
        torch.Tensor
            Kernel value as a scalar tensor.
        """
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        val = self.kernel_circuit(x_np, y_np)
        return torch.tensor(val, dtype=torch.float32)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two sequences of tensors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors where each tensor is of shape (n_wires,).

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        matrix = torch.tensor([[self(x, y) for y in b] for x in a])
        return matrix.detach().cpu().numpy()

__all__ = ["QuantumKernelMethod"]
