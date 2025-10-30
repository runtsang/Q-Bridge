"""Quantum autoencoder implementation using Pennylane.

The QuantumAutoencoder class encapsulates a variational quantum circuit
that maps a classical latent vector to a quantum feature vector.  The
circuit is differentiable via Pennylane's torch interface, enabling
end‑to‑end training of hybrid models.
"""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
import torch
from typing import Iterable, Tuple

class QuantumAutoencoder:
    """Variational quantum circuit that transforms a latent vector into a
    quantum feature vector.

    Parameters
    ----------
    latent_dim : int
        Length of the classical latent vector that will be used as rotation
        angles on the first `latent_dim` qubits.
    num_qubits : int
        Total number of qubits in the device.  Must be >= latent_dim.
    """

    def __init__(self, latent_dim: int, num_qubits: int) -> None:
        if num_qubits < latent_dim:
            raise ValueError("num_qubits must be >= latent_dim")
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits
        self.device = qml.device("default.qubit", wires=num_qubits)
        # QNode with torch interface
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, latent: torch.Tensor) -> torch.Tensor:
        """Variational circuit that uses the latent vector as rotation angles.

        The circuit applies RY rotations on the first `latent_dim` qubits,
        entangles neighboring qubits with CNOT gates, and finally returns
        the expectation values of Pauli‑Z on each qubit.  The output is a
        vector of length `num_qubits` that can be treated as a quantum
        feature vector.
        """
        # Apply RY rotations on the first latent_dim qubits
        for i, angle in enumerate(latent):
            qml.RY(angle, wires=i)
        # Entangle qubits in a simple linear chain
        for i in range(self.num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Return expectation values of Pauli‑Z for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """Apply the quantum circuit to a batch of latent vectors.

        Parameters
        ----------
        latent : torch.Tensor
            Tensor of shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, num_qubits) containing the
            quantum feature vectors for each example in the batch.
        """
        batch_size = latent.shape[0]
        # Apply the QNode element‑wise over the batch
        return torch.stack([self.qnode(latent[i]) for i in range(batch_size)])

__all__ = ["QuantumAutoencoder"]
