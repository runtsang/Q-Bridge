"""
Hybrid autoencoder combining classical MLP with a QCNN feature extractor.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

# Import the QCNN model defined in QCNN.py
from.QCNN import QCNNModel


@dataclass
class HybridAutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class HybridAutoencoderNet(nn.Module):
    """
    A hybrid autoencoder that first maps input data through a QCNN feature
    extractor, then compresses it into a latent vector with a classical
    encoder, and finally reconstructs the input via a classical decoder.
    """

    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.feature_extractor = QCNNModel()

        # Encoder: maps QCNN output (scalar) to latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(1, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.latent_dim),
        )

        # Decoder: reconstructs from latent space to original dimension
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], 1),  # QCNN output is scalar
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return latent representation."""
        feats = self.feature_extractor(inputs)
        return self.encoder(feats)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent representation."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Autoencoder forward pass."""
        return self.decode(self.encode(inputs))


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoderNet:
    """
    Factory that returns a configured HybridAutoencoderNet instance.
    """
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoderNet(config)


def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Train the hybrid autoencoder and return the loss history.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history


def _as_tensor(data: torch.Tensor | torch.Tensor) -> torch.Tensor:
    """Utility: ensure input is a float32 tensor on default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = [
    "HybridAutoencoderConfig",
    "HybridAutoencoderNet",
    "HybridAutoencoder",
    "train_hybrid_autoencoder",
]
