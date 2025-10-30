"""Autoencoder__gen081.py
A robust, configurable autoencoder implementation with training utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, Sequence

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F


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
    """Configuration for the autoencoder architecture and training."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_residual: bool = False      # add skip connections between encoder and decoder
    activation: Callable = nn.ReLU
    dtype: torch.dtype = torch.float32


class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with optional residual connections."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        encoder_layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(cfg.activation())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(cfg.activation())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.encode(x)
        if self.cfg.use_residual:
            # simple skip: concatenate latent with original input before decoding
            x_cat = torch.cat([x, z], dim=-1)
            # adjust decoder input dim accordingly
            if self.decoder[0].in_features!= x_cat.shape[-1]:
                raise RuntimeError("Residual skip requires decoder input dim match.")
            out = self.decoder(x_cat)
        else:
            out = self.decode(z)
        return out


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Sequence[int] | Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    use_residual: bool = False,
    activation: Callable = nn.ReLU,
) -> AutoencoderNet:
    """Factory for a configured autoencoder."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=tuple(hidden_dims),
        dropout=dropout,
        use_residual=use_residual,
        activation=activation,
    )
    return AutoencoderNet(cfg)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    early_stop_patience: int | None = None,
    lr_scheduler: str | None = None,
    device: torch.device | None = None,
) -> list[float]:
    """Train autoencoder with optional early stopping and scheduler."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = None
    if lr_scheduler == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    history: list[float] = []
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        if lr_scheduler and scheduler:
            scheduler.step()

        # early stopping logic
        if early_stop_patience is not None:
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"[Training] Early stopping at epoch {epoch+1}")
                    break

    return history


def evaluate_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> float:
    """Return average reconstruction MSE on the provided data."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    loss_fn = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            total_loss += loss_fn(recon, batch).item() * batch.size(0)

    return total_loss / len(dataset)


def synthetic_data(num_samples: int, dim: int, seed: int | None = None) -> torch.Tensor:
    """Generate random Gaussian data for quick experiments."""
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    return torch.randn(num_samples, dim, generator=rng)


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
    "evaluate_autoencoder",
    "synthetic_data",
]
