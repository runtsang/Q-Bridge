"""Hybrid classical‑quantum autoencoder.

The module implements :class:`HybridAutoencoder`, a PyTorch ``nn.Module`` that
encodes input data into a latent vector, feeds that vector through a Qiskit
quantum circuit (via :class:`QuantumLatentLayer` defined in the companion
``Autoencoder__gen324_qml.py``), and decodes the quantum output back to the input
space.  The design mirrors the classical autoencoder seed but replaces the
latent representation with a variational quantum layer, combining the strengths
of both references."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import the quantum helper from the QML side
from.Autoencoder__gen324_qml import get_quantum_latent_circuit, QuantumLatentLayer


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert a NumPy array, list, or torch tensor into a float32 tensor."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 16
    encoder_hidden: Tuple[int,...] = (128, 64)
    decoder_hidden: Tuple[int,...] = (64, 128)
    dropout: float = 0.1


class HybridAutoencoder(nn.Module):
    """A classical‑quantum autoencoder.

    The encoder maps input data to a latent vector, the quantum latent layer
    transforms this vector via a variational circuit, and the decoder reconstructs
    the data.  The quantum layer is a Qiskit ``SamplerQNN`` wrapped in
    :class:`QuantumLatentLayer`.
    """

    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()

        # Classical encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.encoder_hidden:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum latent layer
        self.quantum_latent = QuantumLatentLayer(
            circuit=get_quantum_latent_circuit(config.latent_dim),
            latent_dim=config.latent_dim,
        )

        # Classical decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in config.decoder_hidden:
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def quantum_transform(self, latent: torch.Tensor) -> torch.Tensor:
        return self.quantum_latent(latent)

    def decode(self, quantum_output: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantum_output)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        q_out = self.quantum_transform(latent)
        return self.decode(q_out)


def hybrid_autoencoder_factory(
    input_dim: int,
    latent_dim: int = 16,
    encoder_hidden: Tuple[int,...] = (128, 64),
    decoder_hidden: Tuple[int,...] = (64, 128),
    dropout: float = 0.1,
) -> HybridAutoencoder:
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        dropout=dropout,
    )
    return HybridAutoencoder(cfg)


def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Reconstruction training loop with MSE loss."""
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
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "hybrid_autoencoder_factory",
    "train_hybrid_autoencoder",
]
