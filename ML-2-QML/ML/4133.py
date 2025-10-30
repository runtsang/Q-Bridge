"""Hybrid autoencoder combining classical FC encoder/decoder with a QCNN‑based quantum encoder."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

# Import the quantum encoder from the corresponding QML module
import Autoencoder__gen110_qml as qml


class HybridAutoencoder(nn.Module):
    """
    Hybrid autoencoder that uses a classical encoder, a QCNN quantum encoder,
    and a classical decoder. Supports configurable hidden layers, dropout,
    and optional quantum latent dimension.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: tuple[int,...] = (128, 64),
        dropout: float = 0.1,
        quantum_latent_dim: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Classical encoder
        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum encoder: map latent_dim -> quantum_latent_dim
        self.quantum_latent_dim = quantum_latent_dim or latent_dim
        self.quantum_encoder = qml.get_quantum_encoder(
            num_qubits=self.quantum_latent_dim,
            latent_dim=self.quantum_latent_dim,
        )

        # Classical decoder
        decoder_layers = []
        in_dim = self.quantum_latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical encoding
        latent = self.encoder(x)
        # Quantum encoding
        latent_np = latent.detach().cpu().numpy()
        quantum_latent = self.quantum_encoder(latent_np)  # shape (batch, quantum_latent_dim)
        quantum_latent_t = torch.tensor(quantum_latent, dtype=x.dtype, device=self.device)
        # Classical decoding
        recon = self.decoder(quantum_latent_t)
        return recon


def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Training loop for the hybrid autoencoder. Only the classical encoder
    and decoder are trained; the quantum circuit remains fixed.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["HybridAutoencoder", "train_hybrid_autoencoder"]
