"""Quantum encoder module for hybrid autoencoder."""

import pennylane as qml
import torch
from pennylane import numpy as np


class HybridAutoencoder:
    """Quantum encoder returning a latent vector using a variational circuit."""

    def __init__(self, latent_dim: int = 8, reps: int = 5):
        self.latent_dim = latent_dim
        self.reps = reps
        self.dev = qml.device("default.qubit", wires=latent_dim, shots=None, interface="torch")
        # Initialize weight parameters for RealAmplitudes
        self.weight_params = torch.nn.Parameter(
            torch.randn(latent_dim, reps, 3, dtype=torch.float32)
        )
        self._qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor, weight_params: torch.Tensor) -> torch.Tensor:
        # Encode classical data into qubits
        qml.AngleEmbedding(x, wires=range(self.latent_dim))
        # Ansatz
        qml.RealAmplitudes(weight_params, wires=range(self.latent_dim))
        # Return expectation values of PauliZ on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.latent_dim)]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent vector for input batch x."""
        return self._qnode(x, self.weight_params)
