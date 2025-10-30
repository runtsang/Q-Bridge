"""Enhanced PyTorch autoencoder with training utilities and reproducibility."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Callable, Optional

import torch
from torch import nn
import torch.optim as optim

# --------------------------------------------------------------------------- #
# 1.  Configuration & utilities
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for the autoencoder network."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU
    bias: bool = True

# --------------------------------------------------------------------------- #
# 2.  Network definition
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """A fully‑connected autoencoder with configurable depth and dropout."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h, bias=cfg.bias))
            encoder_layers.append(cfg.activation())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim, bias=cfg.bias))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h, bias=cfg.bias))
            decoder_layers.append(cfg.activation())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim, bias=cfg.bias))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoding."""
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
# 3.  Factory helper
# --------------------------------------------------------------------------- #
def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU,
    bias: bool = True,
) -> AutoencoderNet:
    """Return a fully‑configured autoencoder."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        bias=bias,
    )
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
# 4.  Training routine
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    loss_fn: nn.Module | None = None,
    early_stopping_patience: int | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> List[float]:
    """
    Train a reconstruction autoencoder.

    Parameters
    ----------
    model : AutoencoderNet
        The network to train.
    data : torch.Tensor
        Input data of shape (N, input_dim).
    epochs : int
        Maximum number of epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization.
    loss_fn : nn.Module | None
        Loss function. Defaults to MSELoss.
    early_stopping_patience : int | None
        If provided, stop training when validation loss does not improve for this
        many epochs. The data is split 80/20 into train/validation.
    device : torch.device | None
        Device to use. Defaults to CUDA if available.
    verbose : bool
        Print progress.

    Returns
    -------
    history : List[float]
        Validation loss history per epoch.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = loss_fn or nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 80/20 split
    n = len(data)
    idx = torch.randperm(n)
    train_idx, val_idx = idx[: int(0.8 * n)], idx[int(0.8 * n) :]
    train_data, val_data = data[train_idx], data[val_idx]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_data),
        batch_size=batch_size,
        shuffle=False,
    )

    history: List[float] = []
    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch, in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch, in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = loss_fn(recon, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_loader.dataset)

        history.append(val_loss)

        if verbose:
            logging.info(f"Epoch {epoch+1:03d} | Train: {epoch_loss:.6f} | Val: {val_loss:.6f}")

        if early_stopping_patience is not None:
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    if verbose:
                        logging.info("Early stopping triggered.")
                    break

    return history

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
]
