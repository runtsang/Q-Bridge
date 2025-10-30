"""Hybrid classical‑quantum autoencoder implementation.

This module defines the AutoencoderGen class that integrates a classical
encoder/decoder with a quantum variational layer implemented using
Pennylane.  The API mirrors the original Autoencoder factory, so
downstream users can simply replace their call to Autoencoder with
AutoencoderGen without changing the rest of their code.

The model is fully differentiable: the quantum layer is wrapped in a
Pennylane QNode with the torch interface, allowing gradients to flow
through the quantum circuit into the classical encoder and decoder.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import pennylane.numpy as np

# Import the quantum layer from the QML module
from quantum_autoencoder import QuantumAutoencoder

@dataclasses.dataclass
class AutoencoderGenConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 8
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    num_qubits: int = 4  # number of qubits in the quantum layer (>= latent_dim)

class AutoencoderGen(nn.Module):
    """Hybrid classical‑quantum autoencoder."""

    def __init__(self, config: AutoencoderGenConfig) -> None:
        super().__init__()
        self.config = config

        # Classical encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum variational layer
        # The latent vector is fed into a quantum circuit that returns a
        # quantum feature vector of dimension `num_qubits`.
        self.quantum_layer = QuantumAutoencoder(
            latent_dim=config.latent_dim,
            num_qubits=config.num_qubits,
        )

        # Classical decoder: maps the quantum feature vector back to the original dimension
        self.decoder = nn.Linear(config.num_qubits, config.input_dim)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode the input into a classical latent vector."""
        return self.encoder(inputs)

    def quantum_forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Apply the quantum variational layer to the latent vector."""
        # The quantum layer expects a 1‑D tensor of shape (latent_dim,)
        # and returns a vector of shape (num_qubits,)
        # We process a batch by applying the QNode element‑wise.
        batch_size = latent.shape[0]
        quantum_features = torch.stack([self.quantum_layer(latent[i]) for i in range(batch_size)])
        return quantum_features

    def decode(self, quantum_features: torch.Tensor) -> torch.Tensor:
        """Decode the quantum features back to the data space."""
        return self.decoder(quantum_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Full forward pass: input -> classical latent -> quantum -> decode."""
        latent = self.encode(inputs)
        quantum_features = self.quantum_forward(latent)
        reconstruction = self.decode(quantum_features)
        return reconstruction

def AutoencoderGenFactory(
    input_dim: int,
    *,
    latent_dim: int = 8,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    num_qubits: int = 4,
) -> AutoencoderGen:
    """Convenience factory that mirrors the original Autoencoder signature."""
    config = AutoencoderGenConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_qubits=num_qubits,
    )
    return AutoencoderGen(config)

def train_autoencoder_gen(
    model: AutoencoderGen,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple training loop for the hybrid autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
        # Optional: early stopping or logging can be added here
    return history

__all__ = [
    "AutoencoderGen",
    "AutoencoderGenConfig",
    "AutoencoderGenFactory",
    "train_autoencoder_gen",
]
