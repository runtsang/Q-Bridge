"""Quantum kernel using Pennylane with trainable ansatz and overlap evaluation."""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
import numpy as np

class QuantumKernelMethod(nn.Module):
    """
    Quantum kernel module that encodes classical data into a parameterised quantum state
    and evaluates the overlap between two encoded states as the kernel value.
    """

    def __init__(self, n_wires: int = 4, layers: int = 2, n_qubits: int | None = None):
        """
        Parameters
        ----------
        n_wires : int
            Number of qubits used for data encoding.
        layers : int
            Depth of the trainable entangling layer.
        n_qubits : int | None
            Explicit number of qubits for the device; defaults to n_wires.
        """
        super().__init__()
        self.n_wires = n_wires
        self.layers = layers
        self.n_qubits = n_qubits or n_wires
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Trainable parameters for the ansatz
        self.params = nn.Parameter(torch.randn(layers, self.n_wires, 3))

        # Prepare a circuit that accepts data and trainable params
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, params: torch.Tensor):
            qml.templates.AngleEmbedding(x, wires=range(self.n_wires))
            qml.templates.StronglyEntanglingLayers(params, wires=range(self.n_wires))
            return qml.state()

        self.circuit = circuit

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value as the absolute overlap between the quantum states
        prepared from x and y using the same trainable parameters.
        """
        # Ensure inputs are 1‑D tensors
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        state_x = self.circuit(x, self.params)
        state_y = self.circuit(y, self.params)

        # Compute overlap
        overlap = torch.abs(torch.dot(state_x.conj(), state_y))
        return overlap

    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
        """Compute Gram matrix between two datasets."""
        K = torch.zeros((len(X), len(Y)))
        for i, xi in enumerate(X):
            for j, yj in enumerate(Y):
                K[i, j] = self.forward(xi, yj)
        return K.detach().numpy()

__all__ = ["QuantumKernelMethod"]
