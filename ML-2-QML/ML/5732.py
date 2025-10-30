"""AutoencoderGen117: classical variational autoencoder with optional KL term."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "AutoencoderGen117",
    "AutoencoderConfig",
    "train_autoencoder",
]


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderGen117`."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    kl_weight: float = 0.0
    """Weight for the KL‑divergence term when training a VAE."""


class AutoencoderGen117(nn.Module):
    """A dense‑layer VAE‑like autoencoder with customisable loss."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_encoder(config)
        self.decoder = self._build_decoder(config)

    # ------------------------------------------------------------------ #
    # Encoder: outputs mean and log‑variance
    # ------------------------------------------------------------------ #
    def _build_encoder(self, cfg: AutoencoderConfig) -> nn.Module:
        layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, cfg.latent_dim * 2))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and log‑variance of the latent distribution."""
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterise the latent vector for back‑propagation."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------ #
    # Decoder: reconstructs the input
    # ------------------------------------------------------------------ #
    def _build_decoder(self, cfg: AutoencoderConfig) -> nn.Module:
        layers: list[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, cfg.input_dim))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from a latent vector."""
        return self.decoder(z)

    # ------------------------------------------------------------------ #
    # Forward and loss
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction, mean, and log‑variance."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss(self, recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute the VAE loss with optional KL weight."""
        recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
        if self.config.kl_weight > 0.0:
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + self.config.kl_weight * kl
        return recon_loss


def train_autoencoder(
    model: AutoencoderGen117,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch)
            loss = model.loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor
