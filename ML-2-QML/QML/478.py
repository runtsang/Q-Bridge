"""Pennylane variational autoencoder with a simple two‑layer ansatz.

The implementation is fully differentiable and can be trained with standard
PyTorch optimizers.  A latent vector is encoded into a subset of qubits and
decoded back to the feature space.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import pennylane as qml
from pennylane import qnode

from typing import Iterable, Tuple


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


class Autoencoder(nn.Module):
    """
    Variational quantum autoencoder implemented with Pennylane.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the classical input.
    latent_dim : int
        Number of qubits used to encode the latent representation.
    dev : str | qml.Device
        Backend device; defaults to the local simulator.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        dev: qml.Device | str = "default.qubit",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Backend
        if isinstance(dev, str):
            self.dev = qml.device(dev, wires=latent_dim + input_dim)
        else:
            self.dev = dev

        # Encoder circuit
        @qnode(self.dev, interface="torch")
        def encoder(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Feature map: rotate each input qubit
            for i in range(self.input_dim):
                qml.RX(x[i], wires=i)
            # Variational layer on latent qubits
            for i in range(self.latent_dim):
                qml.RX(weights[i], wires=self.input_dim + i)
            # Measure latent qubits in Z basis
            return torch.stack(
                [qml.expval(qml.PauliZ(self.input_dim + i)) for i in range(self.latent_dim)]
            )

        # Decoder circuit
        @qnode(self.dev, interface="torch")
        def decoder(latent: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Prepare latent qubits
            for i in range(self.latent_dim):
                qml.RZ(latent[i], wires=self.input_dim + i)
            # Variational layer on input qubits
            for i in range(self.input_dim):
                qml.RZ(weights[i], wires=i)
            # Measure all qubits in X basis to obtain reconstruction
            return torch.stack(
                [qml.expval(qml.PauliX(i)) for i in range(self.input_dim)]
            )

        self.encoder_circuit = encoder
        self.decoder_circuit = decoder

        # Learnable parameters
        self.encoder_weights = nn.Parameter(
            torch.randn(self.latent_dim, dtype=torch.float32)
        )
        self.decoder_weights = nn.Parameter(
            torch.randn(self.input_dim, dtype=torch.float32)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent vector for a batch of inputs."""
        return self.encoder_circuit(x, self.encoder_weights)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Return reconstructed data from latent vector."""
        return self.decoder_circuit(latent, self.decoder_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Autoencoder forward pass."""
        latent = self.encode(x)
        return self.decode(latent)


def train_autoencoder(
    model: Autoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Full training loop for the quantum autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history


__all__ = ["Autoencoder", "train_autoencoder"]
