"""Enhanced fully-connected autoencoder with optional VAE and skip connections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, List

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
    """Configuration for the autoencoder."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    batchnorm: bool = True
    skip_connections: bool = False
    vae: bool = False
    kl_weight: float = 0.001
    device: torch.device | None = None


class AutoencoderNet(nn.Module):
    """A flexible autoencoder that supports VAE, skip connections, and batchnorm."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.to(cfg.device or torch.device("cpu"))

        # Encoder layers
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            if cfg.batchnorm:
                encoder_layers.append(nn.BatchNorm1d(hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        if cfg.vae:
            # Two separate heads for mean and logvar
            self.encoder_mu = nn.Linear(in_dim, cfg.latent_dim)
            self.encoder_logvar = nn.Linear(in_dim, cfg.latent_dim)
        else:
            self.encoder = nn.Sequential(*encoder_layers)
            self.latent_layer = nn.Linear(in_dim, cfg.latent_dim)

        # Decoder layers
        decoder_layers = []
        in_dim = cfg.latent_dim
        hidden_dims_rev = list(reversed(cfg.hidden_dims))
        for hidden in hidden_dims_rev:
            decoder_layers.append(nn.Linear(in_dim, hidden))
            if cfg.batchnorm:
                decoder_layers.append(nn.BatchNorm1d(hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into latent representation."""
        if self.cfg.vae:
            return self.encoder_mu(x)
        return self.encoder(x)

    def sample_latent(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to input space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.cfg.vae:
            h = self.encoder_mu(x)
            logvar = self.encoder_logvar(x)
            z = self.sample_latent(h, logvar)
            recon = self.decode(z)
            return recon, h, logvar
        else:
            z = self.latent_layer(self.encoder(x))
            return self.decode(z)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    val_data: Optional[torch.Tensor] = None,
    early_stop_patience: int = 10,
    kl_weight: Optional[float] = None,
) -> List[float]:
    """Training loop with optional early stopping and VAE KL regularization."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")
    patience = 0
    history: List[float] = []

    train_ds = TensorDataset(_as_tensor(data))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_data is not None:
        val_ds = TensorDataset(_as_tensor(val_data))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch, in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if model.cfg.vae:
                recon, mu, logvar = model(batch)
                recon_loss = loss_fn(recon, batch)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + (kl_weight if kl_weight is not None else model.cfg.kl_weight) * kl_loss
            else:
                recon = model(batch)
                loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        history.append(epoch_loss)

        # Validation & early stopping
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch, in val_loader:
                    batch = batch.to(device)
                    if model.cfg.vae:
                        recon, mu, logvar = model(batch)
                        recon_loss = loss_fn(recon, batch)
                        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + (kl_weight if kl_weight is not None else model.cfg.kl_weight) * kl_loss
                    else:
                        recon = model(batch)
                        loss = loss_fn(recon, batch)
                    val_loss += loss.item() * batch.size(0)
            val_loss /= len(val_loader.dataset)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return history


__all__ = ["AutoencoderNet", "AutoencoderConfig", "train_autoencoder"]
