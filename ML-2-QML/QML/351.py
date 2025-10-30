"""Pennylane variational autoencoder with a linear decoder."""

import pennylane as qml
import torch
from torch import nn

class AutoencoderQNN(nn.Module):
    """Hybrid quantum‑classical autoencoder using Pennylane."""
    def __init__(self, input_dim: int, latent_dim: int = 3, reps: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reps = reps
        self.device = qml.device("default.qubit", wires=latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        # quantum parameters: one per qubit per repetition
        self.params = nn.Parameter(torch.randn(latent_dim * reps))
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.device, interface="torch")
        def circuit(x, params):
            # feature embedding
            qml.templates.AngleEmbedding(features=x, wires=range(self.latent_dim))
            # parameterized layers
            for r in range(self.reps):
                for i in range(self.latent_dim):
                    qml.RY(params[r * self.latent_dim + i], wires=i)
                for i in range(self.latent_dim - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.latent_dim)]
        self.circuit = circuit

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map classical data to a latent vector via the variational circuit."""
        batch = []
        for i in range(x.shape[0]):
            batch.append(self.circuit(x[i], self.params))
        return torch.stack(batch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decoder(z)

def Autoencoder(input_dim: int, latent_dim: int = 3, reps: int = 3) -> AutoencoderQNN:
    """Factory that returns a hybrid quantum‑classical autoencoder."""
    return AutoencoderQNN(input_dim, latent_dim, reps)
