"""Quantum implementation of a variational autoencoder using Pennylane."""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
from typing import List, Tuple


class AutoencoderGen251(nn.Module):
    """Hybrid quantum-classical autoencoder.

    The model uses a parameterised quantum circuit as both encoder and decoder.
    Input data is embedded as rotation angles, the circuit is trained to
    reconstruct the same expectation values.  The latent representation is
    implicitly encoded in the first ``latent_dim`` qubits.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_layers: int = 2,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=input_dim)

        # Parameters for StronglyEntanglingLayers ansatz
        self.params = nn.Parameter(
            torch.randn(num_layers, input_dim, 3, dtype=torch.float32)
        )

        # QNode with autograd support
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, inputs: torch.Tensor, params: torch.Tensor) -> List[torch.Tensor]:
        """Quantum circuit used for encoding and decoding."""
        # Domain wall: flip a contiguous block of qubits (optional)
        for i in range(self.latent_dim, self.input_dim):
            qml.X(i)

        # Angle embedding of the classical data
        qml.AngleEmbedding(inputs, wires=range(self.input_dim))

        # Variational layers
        qml.StronglyEntanglingLayers(params, wires=range(self.input_dim))

        # Return expectation values of Pauli‑Z on all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.input_dim)]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode and immediately decode the input."""
        return self.qnode(inputs, self.params)

    def sample(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        """Generate samples by feeding random latent vectors into the decoder."""
        device = device or torch.device("cpu")
        rnd = torch.randn(batch_size, self.input_dim, device=device)
        return self(rnd)


def train_autoencoder_qml(
    model: AutoencoderGen251,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    lr: float = 0.01,
    device: torch.device | None = None,
) -> List[float]:
    """Training loop for the quantum autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = torch.as_tensor(data, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["AutoencoderGen251", "train_autoencoder_qml"]
