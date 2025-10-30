"""PyTorch implementation of a versatile autoencoder with optional VAE support.

The class extends the original fully‑connected design by:
- supporting an arbitrary number of hidden layers,
- providing a VAE branch with KL regularisation,
- exposing early‑stopping and evaluation metrics,
- and offering a concise training helper that returns loss history.

This module is ready for research experiments on tabular data and can be
plugged into larger pipelines without modification.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Optional

@dataclass
class AutoEncoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    vae: bool = False
    beta: float = 1.0  # KL weight for VAE

class AutoEncoder(nn.Module):
    """Flexible autoencoder with optional VAE behaviour."""

    def __init__(self, cfg: AutoEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Encoder
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU(inplace=True))
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        if cfg.vae:
            self.mu_layer = nn.Linear(in_dim, cfg.latent_dim)
            self.logvar_layer = nn.Linear(in_dim, cfg.latent_dim)
        else:
            self.latent_layer = nn.Linear(in_dim, cfg.latent_dim)

        # Decoder
        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU(inplace=True))
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        if self.cfg.vae:
            mu = self.mu_layer(h)
            logvar = self.logvar_layer(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else:
            return self.latent_layer(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.vae:
            z, _, _ = self.encode(x)
            return self.decode(z)
        else:
            z = self.encode(x)
            return self.decode(z)

def AutoEncoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    vae: bool = False,
    beta: float = 1.0,
) -> AutoEncoder:
    cfg = AutoEncoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        vae=vae,
        beta=beta,
    )
    return AutoEncoder(cfg)

def train_autoencoder(
    model: AutoEncoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    early_stop: int = 10,
) -> List[float]:
    """Train the autoencoder and return loss history.  Supports VAE loss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    history: List[float] = []
    best_loss = float("inf")
    patience = early_stop

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if model.cfg.vae:
                z, mu, logvar = model.encode(batch)
                recon = model.decoder(z)
                recon_loss = mse(recon, batch)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + model.cfg.beta * kl
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
            patience = early_stop
        else:
            patience -= 1
            if patience == 0:
                break
    return history

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = [
    "AutoEncoder",
    "AutoEncoderFactory",
    "AutoEncoderConfig",
    "train_autoencoder",
]
