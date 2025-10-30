"""Hybrid classical auto‑encoder with an optional quantum‑style estimator head.

The architecture is inspired by the original Autoencoder and EstimatorQNN seeds.
It consists of:
* A fully‑connected encoder/decoder with configurable hidden layers.
* A tiny regression head (EstimatorNN) that mirrors the EstimatorQNN network.
* Training utilities that operate purely on classical tensors (PyTorch).

The class is intentionally kept classical to satisfy the constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class HybridAutoEncoderConfig:
    """Configuration for :class:`HybridAutoEncoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    estimator_hidden: Tuple[int, int] = (8, 4)


# --------------------------------------------------------------------------- #
# Core model
# --------------------------------------------------------------------------- #
class HybridAutoEncoder(nn.Module):
    """Classical encoder‑decoder with a latent‑space estimator head."""

    def __init__(self, config: HybridAutoEncoderConfig) -> None:
        super().__init__()

        # Encoder
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

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Estimator head (mirrors EstimatorQNN)
        est_layers = []
        in_dim = config.latent_dim
        for hidden in config.estimator_hidden:
            est_layers.append(nn.Linear(in_dim, hidden))
            est_layers.append(nn.Tanh())
            in_dim = hidden
        est_layers.append(nn.Linear(in_dim, 1))
        self.estimator = nn.Sequential(*est_layers)

    # --------------------------------------------------------------------- #
    # Forward methods
    # --------------------------------------------------------------------- #
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def estimate(self, latents: torch.Tensor) -> torch.Tensor:
        return self.estimator(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss for training the auto‑encoder."""
        latents = self.encode(inputs)
        return self.decode(latents)


# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #
def train_hybrid_autoencoder(
    model: HybridAutoEncoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple MSE training loop for the auto‑encoder part."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = [
    "HybridAutoEncoder",
    "HybridAutoEncoderConfig",
    "train_hybrid_autoencoder",
]
