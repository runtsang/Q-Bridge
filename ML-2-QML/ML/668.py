"""
Autoencoder__gen326.py – Classical VAE implementation.

This module builds on the original fully‑connected autoencoder by adding a
variational layer (mean & log‑variance) and a KL‑divergence penalty.  The
training routine returns a history of both reconstruction and KL losses,
enabling early stopping or monitoring of latent‑space quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Utility helpers
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


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Hyper‑parameters for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    latent_loss_weight: float = 1e-3  # KL‑divergence weight


# --------------------------------------------------------------------------- #
# Core network
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """Variational autoencoder built from linear layers."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Encoder
        encoder_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        # Two heads: mean & log‑var
        self.mean_head = nn.Linear(in_dim, cfg.latent_dim)
        self.logvar_head = nn.Linear(in_dim, cfg.latent_dim)
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, logvar) of latent distribution."""
        h = self.encoder(x)
        return self.mean_head(h), self.logvar_head(h)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    latent_loss_weight: float = 1e-3,
) -> AutoencoderNet:
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        latent_loss_weight=latent_loss_weight,
    )
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# Training routine
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stop_patience: int | None = None,
) -> List[Tuple[float, float]]:
    """
    Train the VAE, returning a list of (recon_loss, kl_loss) tuples.

    Parameters
    ----------
    early_stop_patience : optional int
        Stop training if reconstruction loss does not improve for this many
        consecutive epochs.  Set ``None`` to disable.

    Returns
    -------
    history : list of (recon_loss, kl_loss)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    recon_loss_fn = nn.MSELoss(reduction="sum")
    history: List[Tuple[float, float]] = []

    best_recon = float("inf")
    patience = 0

    for epoch in range(epochs):
        epoch_recon, epoch_kl = 0.0, 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            recon, mean, logvar = model(batch)

            recon_loss = recon_loss_fn(recon, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            loss = recon_loss + model.cfg.latent_loss_weight * kl_loss

            loss.backward()
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

        epoch_recon /= len(dataset)
        epoch_kl /= len(dataset)
        history.append((epoch_recon, epoch_kl))

        # Early stopping on reconstruction loss
        if early_stop_patience is not None:
            if epoch_recon < best_recon:
                best_recon = epoch_recon
                patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    return history


__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
