"""AutoencoderGen377 – an extended classical auto‑encoder with early stopping and latent‑space visualisation.

The class preserves the public API of the original seed but augments it with:
* optional batch‑norm layers after every linear block,
* configurable activation functions,
* an early‑stopping callback based on validation loss,
* a `latent_space` helper that returns the latent representation for a batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Hyper‑parameters for :class:`AutoencoderGen377`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    batch_norm: bool = False
    activation: Callable[[torch.Tensor], torch.Tensor] | None = nn.ReLU()
    early_stop_patience: int | None = 10
    early_stop_delta: float = 1e-4

# --------------------------------------------------------------------------- #
# Core Module
# --------------------------------------------------------------------------- #
class AutoencoderGen377(nn.Module):
    """Fully‑connected auto‑encoder with optional batch‑norm and configurable activations."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        def _block(in_dim: int, out_dim: int) -> nn.Module:
            layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            if cfg.activation is not None:
                layers.append(cfg.activation)
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            return nn.Sequential(*layers)

        # Encoder
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(_block(in_dim, h))
            in_dim = h
        enc_layers.append(_block(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(_block(in_dim, h))
            in_dim = h
        dec_layers.append(_block(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    # --------------------------------------------------------------------- #
    # Forward, encode, decode helpers
    # --------------------------------------------------------------------- #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

    # --------------------------------------------------------------------- #
    # Latent‑space visualisation
    # --------------------------------------------------------------------- #
    def latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent embeddings for ``x``."""
        return self.encode(x)

    def plot_latent(self, x: torch.Tensor, labels: Optional[Iterable[int]] = None) -> None:
        """Simple 2‑D scatter plot of the latent space when latent_dim == 2."""
        if self.cfg.latent_dim!= 2:
            raise ValueError("latent_dim must be 2 for plotting")
        z = self.encode(x).detach().cpu().numpy()
        plt.figure(figsize=(6, 6))
        if labels is None:
            plt.scatter(z[:, 0], z[:, 1], alpha=0.7, s=15)
        else:
            for l in set(labels):
                idx = [i for i, lbl in enumerate(labels) if lbl == l]
                plt.scatter(z[idx, 0], z[idx, 1], label=str(l), alpha=0.7, s=15)
            plt.legend()
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.title("Latent space")
        plt.grid(True)
        plt.show()

# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

def train_autoencoder(
    model: AutoencoderGen377,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    val_split: float = 0.1,
    shuffle: bool = True,
) -> tuple[list[float], list[float]]:
    """Extended training loop – returns (train_loss_hist, val_loss_hist)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Split data into train / val
    n = len(data)
    idx = torch.randperm(n)
    split = int(n * (1 - val_split))
    train_idx, val_idx = idx[:split], idx[split:]
    train_loader = DataLoader(TensorDataset(_as_tensor(data[train_idx])),
                              batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(TensorDataset(_as_tensor(data[val_idx])),
                            batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    val_history: list[float] = []

    patience = model.cfg.early_stop_patience
    delta = model.cfg.early_stop_delta
    best_val = float("inf")
    counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch, in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(train_loader.dataset)
        history.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch, in val_loader:
                batch = batch[0].to(device)
                recon = model(batch)
                loss = loss_fn(recon, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_history.append(val_loss)

        # Early stopping
        if patience is not None and val_loss < best_val - delta:
            best_val = val_loss
            counter = 0
        else:
            counter += 1

        if patience is not None and counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train loss: {epoch_loss:.4f} | Val loss: {val_loss:.4f}")

    return history, val_history

# --------------------------------------------------------------------------- #
# Helper factory
# --------------------------------------------------------------------------- #
def AutoencoderGen377_factory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    batch_norm: bool = False,
    activation: Callable[[torch.Tensor], torch.Tensor] | None = nn.ReLU(),
    early_stop_patience: int | None = 10,
    early_stop_delta: float = 1e-4,
) -> AutoencoderGen377:
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        batch_norm=batch_norm,
        activation=activation,
        early_stop_patience=early_stop_patience,
        early_stop_delta=early_stop_delta,
    )
    return AutoencoderGen377(cfg)

__all__ = [
    "AutoencoderGen377",
    "AutoencoderConfig",
    "train_autoencoder",
    "AutoencoderGen377_factory",
]
