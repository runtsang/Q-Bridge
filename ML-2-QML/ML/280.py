"""AutoencoderGen311: a versatile fully‑connected autoencoder with optional KL regularisation and early‑stopping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU
    use_vae: bool = False
    kl_weight: float = 1e-3

class AutoencoderNet(nn.Module):
    """A fully‑connected (variational) autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_mlp(
            in_dim=config.input_dim,
            out_dim=2 * config.latent_dim if config.use_vae else config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            activation=config.activation,
        )
        if config.use_vae:
            # split into μ and logσ
            self.mu_layer = nn.Linear(2 * config.latent_dim, config.latent_dim)
            self.logvar_layer = nn.Linear(2 * config.latent_dim, config.latent_dim)
        self.decoder = self._build_mlp(
            in_dim=config.latent_dim,
            out_dim=config.input_dim,
            hidden_dims=config.hidden_dims[::-1],
            dropout=config.dropout,
            activation=config.activation,
        )

    def _build_mlp(self, in_dim: int, out_dim: int,
                   hidden_dims: Tuple[int,...],
                   dropout: float,
                   activation: Callable[[torch.Tensor], torch.Tensor]) -> nn.Sequential:
        layers = []
        dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            dim = h
        layers.append(nn.Linear(dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        if self.config.use_vae:
            mu, logvar = h.chunk(2, dim=-1)
            return mu, logvar
        return h

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
            return self.decode(z), mu, logvar
        else:
            return self.decode(self.encode(x))

def AutoencoderGen311(input_dim: int,
                      *,
                      latent_dim: int = 32,
                      hidden_dims: Tuple[int,...] = (128, 64),
                      dropout: float = 0.1,
                      activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU,
                      use_vae: bool = False,
                      kl_weight: float = 1e-3) -> AutoencoderNet:
    """Factory that returns a configured autoencoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        use_vae=use_vae,
        kl_weight=kl_weight,
    )
    return AutoencoderNet(config)

def train_autoencoder(model: AutoencoderNet,
                      data: torch.Tensor,
                      *,
                      epochs: int = 200,
                      batch_size: int = 128,
                      lr: float = 1e-3,
                      weight_decay: float = 0.0,
                      device: torch.device | None = None,
                      early_stop_patience: Optional[int] = None,
                      callback: Optional[Callable[[int, float], None]] = None) -> list[float]:
    """Training loop that supports VAE loss and early stopping."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss(reduction="sum")
    history: list[float] = []
    best_loss = float("inf")
    patience = early_stop_patience or 0
    counter = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if model.config.use_vae:
                recon, mu, logvar = model(batch)
                recon_loss = mse_loss(recon, batch)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + model.config.kl_weight * kl
            else:
                recon = model(batch)
                loss = mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        if callback:
            callback(epoch, epoch_loss)

        if early_stop_patience is not None:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                counter = 0
            else:
                counter += 1
                if counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    return history

__all__ = ["AutoencoderGen311", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
