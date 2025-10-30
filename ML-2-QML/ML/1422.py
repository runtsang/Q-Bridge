"""Enhanced classical autoencoder with optional VAE, batchnorm, dropout, and early stopping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

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
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderHybrid`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_vae: bool = False
    batchnorm: bool = False


class AutoencoderHybrid(nn.Module):
    """Classical autoencoder with optional variational component."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        if config.use_vae:
            # Additional linear layers to produce mean and log‑variance
            self.mu_layer = nn.Linear(config.latent_dim, config.latent_dim)
            self.logvar_layer = nn.Linear(config.latent_dim, config.latent_dim)

    def _build_encoder(self) -> nn.Sequential:
        layers = []
        in_dim = self.config.input_dim
        for h in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if self.config.batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.config.latent_dim))
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        layers = []
        in_dim = self.config.latent_dim
        for h in reversed(self.config.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            if self.config.batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.config.input_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        if self.config.use_vae:
            mu = self.mu_layer(latent)
            logvar = self.logvar_layer(latent)
            return mu, logvar
        return latent

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.use_vae:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z)
        else:
            return self.decode(self.encode(x))

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence between latent distribution and standard normal."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def reconstruction_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(recon, target, reduction="mean")


def train_autoencoder(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stop_patience: int | None = None,
) -> list[float]:
    """Training loop that supports VAE and early stopping."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            if model.config.use_vae:
                mu, logvar = model.encode(batch)
                z = model.reparameterize(mu, logvar)
                recon = model.decode(z)
                loss = model.reconstruction_loss(recon, batch) + model.kl_loss(mu, logvar)
            else:
                recon = model(batch)
                loss = model.reconstruction_loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        # Early stopping logic
        if early_stop_patience is not None:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break
    return history


__all__ = ["AutoencoderHybrid", "AutoencoderConfig", "train_autoencoder"]
