"""
AutoencoderV2: A fully‑connected autoencoder with configurable architecture and training utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler

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
class AutoencoderV2Config:
    """Configuration for :class:`AutoencoderV2`."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.0
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()
    use_batch_norm: bool = False
    device: torch.device | None = None


class AutoencoderV2(nn.Module):
    """An autoencoder with explicit encoder/decoder modules and optional batch‑norm."""

    def __init__(self, cfg: AutoencoderV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Encoder
        enc_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            if cfg.use_batch_norm:
                enc_layers.append(nn.BatchNorm1d(hidden))
            enc_layers.append(cfg.activation)
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden))
            if cfg.use_batch_norm:
                dec_layers.append(nn.BatchNorm1d(hidden))
            dec_layers.append(cfg.activation)
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def sample_latent(self, shape: Tuple[int,...]) -> torch.Tensor:
        """Sample from a standard normal prior."""
        return torch.randn(shape, dtype=torch.float32, device=self.device)


def train_autoencoder_v2(
    model: AutoencoderV2,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    early_stop: int | None = 20,
    scheduler: Optional[_LRScheduler] = None,
    device: torch.device | None = None,
) -> list[float]:
    """Train an AutoencoderV2 with optional early stopping and scheduler.

    Returns a list of training losses per epoch.
    """
    device = device or model.device
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer: Optimizer = Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()
    history: list[float] = []

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        if scheduler is not None:
            scheduler.step(epoch_loss)

        # Early‑stopping logic
        if early_stop is not None and epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if early_stop is not None and epochs_no_improve >= early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return history


__all__ = [
    "AutoencoderV2",
    "AutoencoderV2Config",
    "train_autoencoder_v2",
]
