"""Quantum kernel via variational circuit with trainable parameters."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml
import torch
from torch import nn

class QuantumKernelMethod(nn.Module):
    """
    Quantum kernel using a variational circuit.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits in the device.
    n_layers : int, default 2
        Number of variational layers.
    device_name : str, default "default.qubit"
        PennyLane device name.
    shots : int, default 1024
        Number of shots for state preparation (used only when the
        device requires sampling).

    Notes
    -----
    The kernel value between two classical vectors *x* and *y* is defined as
    the squared magnitude of the inner product of the quantum states
    prepared by encoding *x* and *y* into the same variational ansatz.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        device_name: str = "default.qubit",
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.dev = qml.device(device_name, wires=n_qubits, shots=shots)
        # Variational parameters: shape (n_layers, n_qubits, 3)
        self.params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3, dtype=torch.float64)
        )

    def _state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the state vector prepared by encoding *x*.
        """
        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            # Input encoding (RY rotations)
            for i, val in enumerate(x):
                qml.RY(val, wires=i)
            # Variational layers
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.Rot(*self.params[layer, qubit], wires=qubit)
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            return qml.state()
        return circuit(x)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel between two batches of samples.

        Parameters
        ----------
        x : torch.Tensor, shape (m, d) or (1, d)
        y : torch.Tensor, shape (n, d) or (1, d)

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (m, n).
        """
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        m, n = x.shape[0], y.shape[0]
        K = torch.zeros((m, n), dtype=torch.float64)
        for i in range(m):
            state_x = self._state(x[i])
            for j in range(n):
                state_y = self._state(y[j])
                # Fidelity: |<state_x | state_y>|^2
                kernel_val = torch.abs(torch.vdot(state_x, state_y)) ** 2
                K[i, j] = kernel_val
        return K

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two sequences of tensors.

        Parameters
        ----------
        a : Sequence[torch.Tensor]
        b : Sequence[torch.Tensor]

        Returns
        -------
        np.ndarray
            NumPy array of shape (len(a), len(b)).
        """
        m, n = len(a), len(b)
        K = torch.zeros((m, n), dtype=torch.float64)
        for i in range(m):
            for j in range(n):
                K[i, j] = self.forward(a[i], b[j]).item()
        return K.cpu().numpy()

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.01,
        epochs: int = 100,
    ) -> None:
        """
        Simple training loop that optimises the variational parameters
        by maximizing the kernel alignment with a linear SVM.

        Parameters
        ----------
        X : torch.Tensor
            Training data of shape (N, d).
        y : torch.Tensor
            Binary labels of shape (N,) with values ±1.
        lr : float, default 0.01
            Learning rate for the optimiser.
        epochs : int, default 100
            Number of optimisation steps.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            K = self.forward(X, X)
            # Kernel alignment loss (negative because we maximise it)
            loss = - (y @ K @ y) / (y @ y)
            loss.backward()
            optimizer.step()

__all__ = ["QuantumKernelMethod"]
