"""AutoencoderGen318: Variational quantum autoencoder using Pennylane."""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn

__all__ = ["AutoencoderGen318"]


class AutoencoderGen318(nn.Module):
    """
    Variational quantum autoencoder.

    The encoder is a parameterised quantum circuit that maps an input vector
    to a latent vector via expectation values of Pauli‑Z operators.
    A classical linear decoder reconstructs the input.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 4,
        reps: int = 3,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reps = reps
        self.dev = qml.device("default.qubit", wires=latent_dim)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, weights):
            # Feature encoding
            for i, val in enumerate(x):
                qml.RX(val, wires=i % latent_dim)
            # Variational block
            qml.templates.BasicEntanglerLayers(weights=weights, wires=range(latent_dim))
            # Output expectation values
            return [qml.expval(qml.PauliZ(w)) for w in range(latent_dim)]

        self.circuit = circuit
        self.weights = nn.Parameter(torch.randn(reps, latent_dim))

        # Classical decoder
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map input to latent representation using the quantum circuit."""
        return self.circuit(x, self.weights)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from the latent vector."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass."""
        z = self.encode(x)
        return self.decode(z)

    def loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean‑squared‑error loss."""
        return torch.mean((recon - target) ** 2)

    def train_step(
        self,
        data: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        *,
        lr: float | None = None,
    ) -> torch.Tensor:
        """Perform one training step and return the loss."""
        optimizer.zero_grad()
        recon = self(data)
        loss = self.loss(recon, data)
        loss.backward()
        optimizer.step()
        return loss
