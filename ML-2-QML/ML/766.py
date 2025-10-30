"""Enhanced PyTorch denoising autoencoder with noise schedule and KL regularisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, Optional

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
    """Configuration for the denoising autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    noise_schedule: Callable[[int], float] = lambda _: 0.5
    """Callable returning the noise level for each epoch."""


class AutoencoderGen(nn.Module):
    """A variational denoising autoencoder with optional KL regularisation."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder network
        encoder_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent mean and log‑variance
        self.mu_layer = nn.Linear(in_dim, config.latent_dim)
        self.logvar_layer = nn.Linear(in_dim, config.latent_dim)

        # Decoder network
        decoder_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar


def _add_noise(x: torch.Tensor, noise_level: float) -> torch.Tensor:
    """Corrupt input with Gaussian noise of given standard deviation."""
    return x + noise_level * torch.randn_like(x)


def _mse_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(recon, target, reduction="mean")


def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_autoencoder(
    model: AutoencoderGen,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    kl_weight: float = 1.0,
    device: torch.device | None = None,
) -> list[float]:
    """Standard reconstruction training loop for the VAE autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = _mse_loss(recon, batch) + kl_weight * _kl_loss(mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def train_denoising_autoencoder(
    model: AutoencoderGen,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    kl_weight: float = 1.0,
    noise_schedule: Callable[[int], float] | None = None,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop that corrupts inputs each epoch according to a schedule."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    noise_schedule = noise_schedule or (lambda epoch: 0.5)
    for epoch in range(epochs):
        epoch_loss = 0.0
        noise_level = noise_schedule(epoch)
        for (batch,) in loader:
            batch = batch.to(device)
            noisy_batch = _add_noise(batch, noise_level)
            optimizer.zero_grad()
            recon, mu, logvar = model(noisy_batch)
            loss = _mse_loss(recon, batch) + kl_weight * _kl_loss(mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "AutoencoderGen",
    "AutoencoderConfig",
    "train_autoencoder",
    "train_denoising_autoencoder",
]
