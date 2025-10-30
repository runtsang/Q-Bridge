"""Quantum-only autoencoder implemented with PennyLane."""

import pennylane as qml
import pennylane.numpy as npq
import torch
from torch import nn

__all__ = ["QuantumAutoencoder"]


class QuantumAutoencoder(nn.Module):
    """Pure quantum autoencoder using a variational circuit.

    The class expects the input dimension to match the latent dimension,
    because the latent vector is encoded directly into the qubits.
    """

    def __init__(self, input_dim: int, latent_dim: int, reps: int = 3):
        super().__init__()
        if input_dim!= latent_dim:
            raise ValueError(
                "QuantumAutoencoder requires input_dim == latent_dim for direct angle encoding."
            )
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reps = reps

        # Pennylane device
        self.device = qml.device("default.qubit", wires=latent_dim)

        # Define the QNode
        def _qnode(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Angle embedding of the input vector
            qml.templates.AngleEmbedding(x, wires=range(latent_dim))
            # Variational layer
            qml.templates.RealAmplitudes(params, wires=range(latent_dim), reps=reps)
            # Return expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(latent_dim)]

        self.qnode = qml.QNode(_qnode, self.device, interface="torch")

        # Number of variational parameters
        self.num_params = reps * latent_dim * 2  # RealAmplitudes uses 2 parameters per qubit per layer
        self.qparams = nn.Parameter(torch.randn(self.num_params))

        # Classical linear decoder to map back to input space
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        q_outputs = []
        for i in range(batch_size):
            out = self.qnode(x[i], self.qparams)
            q_outputs.append(out)
        q_outputs = torch.stack(q_outputs)
        return self.decoder(q_outputs)
