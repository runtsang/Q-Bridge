"""Hybrid classical-quantum autoencoder combining MLP and convolutional filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------
# Classical convolutional filter (classical emulation of a quantum filter)
# ----------------------------------------------------------------------
class ConvFilter(nn.Module):
    """Emulates a quantum convolutional filter with a classical 2D Conv layer."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

# ----------------------------------------------------------------------
# Hybrid autoencoder configuration
# ----------------------------------------------------------------------
@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    kernel_size: int = 2

# ----------------------------------------------------------------------
# Hybrid autoencoder network
# ----------------------------------------------------------------------
class HybridAutoencoderNet(nn.Module):
    """A hybrid autoencoder that first applies a convolutional filter, then an MLP encoder/decoder."""

    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.conv_filter = ConvFilter(kernel_size=config.kernel_size)

        # Compute flattened size after convolution
        conv_output_dim = config.input_dim - config.kernel_size + 1
        flat_dim = conv_output_dim * conv_output_dim

        # Encoder
        in_dim = flat_dim
        encoder_layers = []
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
        decoder_layers.append(nn.Linear(in_dim, flat_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv_filter(inputs)
        flat = conv_out.view(conv_out.size(0), -1)
        return self.encoder(flat)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(latents)
        batch_size = latents.size(0)
        side = int((flat.size(1)) ** 0.5)
        return flat.view(batch_size, 1, side, side)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    kernel_size: int = 2,
) -> HybridAutoencoderNet:
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        kernel_size=kernel_size,
    )
    return HybridAutoencoderNet(config)

# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
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
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data)
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

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "HybridAutoencoderNet",
    "train_hybrid_autoencoder",
    "ConvFilter",
]
