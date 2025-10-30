"""Quantum latent layer for hybrid autoencoder.

This module uses Pennylane to define a parameterised circuit that acts
as a latent regularizer.  The circuit receives a classical latent vector,
encodes it into qubit angles, applies a trainable rotation layer, and
returns the expectation values of Pauli‑Z on each qubit.  The returned
vector has the same dimensionality as the number of qubits and is fed
back into the classical decoder.
"""

import pennylane as qml
import torch
from torch import nn


class QuantumLatentLayer(nn.Module):
    """Parameterised quantum circuit that transforms a latent vector.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the classical latent representation.
    num_qubits : int
        Number of qubits used for the quantum transformation.
    """

    def __init__(self, latent_dim: int, num_qubits: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits
        # Device – use the default simulator; in practice a real backend
        # can be swapped in without changing the interface.
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        # QNode that implements the circuit
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

        # Trainable parameters for a single variational layer
        self.params = nn.Parameter(
            torch.randn(self.num_qubits, 3, dtype=torch.float32)
        )

        # Linear map from the classical latent space to the qubit register
        self.lin = nn.Linear(self.latent_dim, self.num_qubits)

    def _circuit(self, latent: torch.Tensor, params: torch.Tensor) -> list[torch.Tensor]:
        """
        Circuit:

        * Angle‑encoding of the latent vector using RY gates.
        * One layer of trainable rotations (RX, RY, RZ).
        * Return the expectation value of Pauli‑Z on each qubit.
        """
        # Encode latent into qubits
        for i, val in enumerate(latent):
            qml.RY(val, wires=i)

        # Variational rotations
        for i in range(self.num_qubits):
            qml.Rot(params[i, 0], params[i, 1], params[i, 2], wires=i)

        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        latent : torch.Tensor
            Shape (batch, latent_dim)

        Returns
        -------
        torch.Tensor
            Shape (batch, num_qubits) – the regularised latent representation.
        """
        # Project to the qubit register
        projected = self.lin(latent)  # (batch, num_qubits)
        # Execute the quantum circuit
        return self.qnode(projected, self.params)


__all__ = ["QuantumLatentLayer"]
