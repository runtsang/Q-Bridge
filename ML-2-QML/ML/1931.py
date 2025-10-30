"""AutoencoderGen119 – a PyTorch Variational Autoencoder with flexible loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterable, Callable

import torch
from torch import nn
import torch.nn.functional as F
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
class AutoencoderGen119Config:
    """Configuration for :class:`AutoencoderGen119`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    kl_weight: float = 0.001  # weight for KL divergence
    loss_fn: str = "mse"  #'mse' or 'bce'


class AutoencoderGen119(nn.Module):
    """A Variational Autoencoder with configurable architecture."""

    def __init__(self, cfg: AutoencoderGen119Config) -> None:
        super().__init__()
        self.cfg = cfg

        # Encoder
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # Mean & log‑var heads
        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

        # Decoder
        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def loss_function(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    cfg: AutoencoderGen119Config,
) -> torch.Tensor:
    """Compute reconstruction + KL divergence."""
    if cfg.loss_fn.lower() == "bce":
        recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    else:
        recon_loss = F.mse_loss(recon, x, reduction="sum")

    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + cfg.kl_weight * kld


def train_autoencoder_gen119(
    model: AutoencoderGen119,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop returning loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch)
            loss = loss_function(recon, batch, mu, logvar, model.cfg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        history.append(epoch_loss / len(dataset))
    return history


def AutoencoderGen119(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    kl_weight: float = 0.001,
    loss_fn: str = "mse",
) -> AutoencoderGen119:
    """Convenience factory mirroring the original API."""
    cfg = AutoencoderGen119Config(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        kl_weight=kl_weight,
        loss_fn=loss_fn,
    )
    return AutoencoderGen119(cfg)


__all__ = [
    "AutoencoderGen119",
    "AutoencoderGen119Config",
    "AutoencoderGen119",
    "train_autoencoder_gen119",
    "loss_function",
]
