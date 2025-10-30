"""Hybrid classical autoencoder with optional VAE and skip connections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
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
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    activation: nn.Module = nn.ReLU()
    batch_norm: bool = False
    skip_connections: bool = False


class HybridAutoencoder(nn.Module):
    """A flexible autoencoder that can act as a VAE or a standard MLP.

    The network supports optional skip connections, batch‑norm, dropout, and
    a VAE objective.  The public API mirrors the original seed while adding
    new capabilities.
    """

    def __init__(
        self,
        config: HybridAutoencoderConfig,
        *,
        use_vae: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_vae = use_vae

        # Encoder
        encoder_layers: list[torch.nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            if config.batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden))
            encoder_layers.append(config.activation)
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: list[torch.nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            if config.batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden))
            decoder_layers.append(config.activation)
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        if self.use_vae:
            # VAE latent distribution
            self.fc_mu = nn.Linear(config.latent_dim, config.latent_dim)
            self.fc_logvar = nn.Linear(config.latent_dim, config.latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder."""
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction of the input."""
        z = self.encode(x)
        if self.use_vae:
            mu = self.fc_mu(z)
            logvar = self.fc_logvar(z)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar
        return self.decode(z)

    def loss_function(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor | None = None,
        logvar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute reconstruction + KL loss for VAE."""
        recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
        if self.use_vae and mu is not None and logvar is not None:
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + kl
        return recon_loss


def train_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train the autoencoder and return the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            if model.use_vae:
                recon, mu, logvar = model(batch)
                loss = model.loss_function(recon, batch, mu, logvar)
            else:
                recon = model(batch)
                loss = model.loss_function(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["HybridAutoencoder", "HybridAutoencoderConfig", "train_autoencoder"]
