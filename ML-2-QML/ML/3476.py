"""Hybrid classifier with NumPy‑based quantum simulation.

This module defines `HybridFCL`, a PyTorch `nn.Module` that
combines:
  * a classical encoder mapping raw features to rotation angles,
  * a variational quantum circuit simulated with NumPy,
  * a linear readout producing class logits.

The implementation is fully classical and therefore can be used when
a quantum backend is unavailable or for quick benchmarking.
"""

import numpy as np
import torch
from torch import nn

class HybridFCL(nn.Module):
    """
    Hybrid classifier that simulates a parameterised quantum circuit
    with NumPy and uses the resulting expectation values as
    features for a linear output layer.

    Parameters
    ----------
    num_features : int
        Number of input features.
    n_qubits : int
        Number of qubits in the simulated circuit.
    depth : int
        Number of variational layers.
    num_classes : int
        Number of target classes.
    """

    def __init__(self, num_features: int, n_qubits: int, depth: int, num_classes: int):
        super().__init__()

        self.n_qubits = n_qubits
        self.depth = depth

        # Encoder: maps raw features → rotation angles for the encoding gates
        self.encoder = nn.Linear(num_features, n_qubits)

        # Variational parameters (weights) of the quantum circuit:
        # one Ry per qubit per depth layer
        self.theta = nn.Parameter(torch.randn(n_qubits * depth))

        # Linear head for classification
        self.classifier = nn.Linear(n_qubits, num_classes)

    def _quantum_expectation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute expectation values of Z on each qubit for a batch of inputs.

        Parameters
        ----------
        x : torch.Tensor
            Batch of encoding angles of shape (batch, n_qubits).

        Returns
        -------
        torch.Tensor
            Expectation values of shape (batch, n_qubits).
        """
        x_np = x.detach().cpu().numpy()
        theta_np = self.theta.detach().cpu().numpy()

        batch = x_np.shape[0]
        exp_vals = np.zeros((batch, self.n_qubits), dtype=np.float32)

        for i in range(batch):
            # Initialise state |0...0>
            dim = 2 ** self.n_qubits
            state = np.zeros(dim, dtype=np.complex128)
            state[0] = 1.0

            # Encoding Rx gates
            for q, angle in enumerate(x_np[i]):
                Rx = np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)],
                               [-1j * np.sin(angle / 2), np.cos(angle / 2)]])
                U = np.eye(1, dtype=np.complex128)
                for k in range(self.n_qubits):
                    U = np.kron(U, Rx if k == q else np.eye(2, dtype=np.complex128))
                state = U @ state

            # Variational layers
            idx = 0
            for _ in range(self.depth):
                for q in range(self.n_qubits):
                    theta = theta_np[idx]
                    idx += 1
                    Ry = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                                   [np.sin(theta / 2), np.cos(theta / 2)]])
                    U = np.eye(1, dtype=np.complex128)
                    for k in range(self.n_qubits):
                        U = np.kron(U, Ry if k == q else np.eye(2, dtype=np.complex128))
                    state = U @ state
                # CZ gates between neighbours (skip entanglement for brevity)
                # In a real implementation, add CZ unitaries here.

            # Expectation of Z on each qubit
            for q in range(self.n_qubits):
                exp = 0.0
                for idx, amp in enumerate(state):
                    bit = (idx >> q) & 1
                    prob = np.abs(amp) ** 2
                    exp += (1.0 if bit == 0 else -1.0) * prob
                exp_vals[i, q] = exp

        return torch.from_numpy(exp_vals).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        encoded = self.encoder(x)
        quantum_features = self._quantum_expectation(encoded)
        logits = self.classifier(quantum_features)
        return logits

__all__ = ["HybridFCL"]
