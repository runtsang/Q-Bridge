"""PyTorch implementation of an enhanced autoencoder with optional VAE functionality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Coerce input into a float32 tensor."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderGen018Config:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    batchnorm: bool = False
    vaf: bool = False  # Variational Autoencoder flag


class AutoencoderGen018(nn.Module):
    def __init__(self, config: AutoencoderGen018Config) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_mlp(
            config.input_dim,
            config.hidden_dims,
            out_features=config.latent_dim,
            batchnorm=config.batchnorm,
            dropout=config.dropout,
        )
        if config.vaf:
            # VAE needs separate mu and logvar layers
            self.mu_layer = nn.Linear(config.latent_dim, config.latent_dim)
            self.logvar_layer = nn.Linear(config.latent_dim, config.latent_dim)
        self.decoder = self._build_mlp(
            config.latent_dim,
            tuple(reversed(config.hidden_dims)),
            out_features=config.input_dim,
            batchnorm=config.batchnorm,
            dropout=config.dropout,
        )

    def _build_mlp(
        self,
        in_dim: int,
        hidden_dims: Tuple[int,...],
        out_features: int,
        batchnorm: bool,
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_features))
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(inputs)
        if self.config.vaf:
            mu = self.mu_layer(z)
            logvar = self.logvar_layer(z)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        return z

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.vaf:
            z, mu, logvar = self.encode(inputs)
            recon = self.decode(z)
            return recon, mu, logvar
        else:
            return self.decode(self.encode(inputs))


def train_autoencoder(
    model: AutoencoderGen018,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    early_stop_patience: int = 10,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    history: list[float] = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if model.config.vaf:
                recon, mu, logvar = model(batch)
                recon_loss = mse(recon, batch)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl
            else:
                recon = model(batch)
                loss = mse(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    return history


__all__ = [
    "AutoencoderGen018",
    "AutoencoderGen018Config",
    "train_autoencoder",
]
