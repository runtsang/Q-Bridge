"""Quantum kernel using a neural‑network controlled feature map and a SWAP test.

The kernel value is the fidelity between two encoded states obtained by
feeding the input vectors through a small neural network that produces
rotation angles for the RY gates.  The fidelity is estimated with a
single‑ancilla SWAP test on a PennyLane simulator.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
import pennylane as qml

__all__ = ["QuantumKernelMethod", "kernel_matrix"]


class QuantumKernelMethod(nn.Module):
    """Quantum kernel with a learnable feature map.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the classical input vectors.
    hidden_dim : int, optional
        Hidden layer size of the feature‑map network.
    device : str or torch.device, optional
        Target device for the neural network. Defaults to ``'cpu'``.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16, device: str | torch.device = "cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)

        # Neural network that maps a classical vector to rotation angles.
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        ).to(self.device)

        # PennyLane device: 2*input_dim data qubits + 1 ancilla.
        self.qdevice = qml.device("default.qubit", wires=2 * input_dim + 1, shots=0)

        input_dim_local = self.input_dim

        @qml.qnode(self.qdevice, interface="torch", diff_method="backprop")
        def _qnode(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Encode x on the first half of qubits.
            angles_x = self.feature_net(x)
            for i in range(input_dim_local):
                qml.RY(angles_x[i], wires=i)

            # Encode y on the second half of qubits.
            angles_y = self.feature_net(y)
            for i in range(input_dim_local):
                qml.RY(angles_y[i], wires=input_dim_local + i)

            # SWAP test using an ancilla qubit.
            ancilla = 2 * input_dim_local
            qml.Hadamard(wires=ancilla)
            for i in range(input_dim_local):
                qml.CSWAP(wires=[ancilla, i, input_dim_local + i])
            qml.Hadamard(wires=ancilla)

            # Return the expectation of PauliZ on the ancilla.
            return qml.expval(qml.PauliZ(wires=ancilla))

        self._qnode = _qnode

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value for two input vectors.

        The kernel is defined as (1 + <Z>)/2 where <Z> is the expectation
        value of PauliZ on the ancilla after the SWAP test.
        """
        x = x.to(self.device)
        y = y.to(self.device)
        z = self._qnode(x, y)
        return (1 + z) / 2


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], input_dim: int, hidden_dim: int = 16, device: str | torch.device = "cpu") -> np.ndarray:
    """
    Compute the Gram matrix between two collections of vectors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of 1‑D tensors that must all have the same length
        ``input_dim``.
    input_dim : int
        Dimensionality of the input vectors.
    hidden_dim : int, optional
        Hidden layer size of the feature‑map network.
    device : str or torch.device, optional
        Target device for the neural network.
    """
    kernel = QuantumKernelMethod(input_dim, hidden_dim, device)
    return np.array([[kernel(x, y).item() for y in b] for x in a])
