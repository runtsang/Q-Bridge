"""Enhanced fully‑connected autoencoder with configurable regularization, batchnorm, and early stopping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_
import math


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
    """Configuration for :class:`Autoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_batchnorm: bool = False
    weight_decay: float = 0.0
    max_norm: Optional[float] = None  # for gradient clipping


class Autoencoder(nn.Module):
    """A flexible MLP autoencoder with optional batch‑norm and gradient clipping."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_block(
            in_dim=config.input_dim,
            dims=config.hidden_dims,
            out_dim=config.latent_dim,
            name="encoder",
        )
        self.decoder = self._build_block(
            in_dim=config.latent_dim,
            dims=tuple(reversed(config.hidden_dims)),
            out_dim=config.input_dim,
            name="decoder",
        )

    def _build_block(
        self,
        in_dim: int,
        dims: Tuple[int,...],
        out_dim: int,
        name: str,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        current = in_dim
        for hidden in dims:
            layers.append(nn.Linear(current, hidden))
            if self.config.use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            current = hidden
        layers.append(nn.Linear(current, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def train_autoencoder(
    model: Autoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stop_patience: int = 10,
    max_grad_norm: Optional[float] = None,
) -> List[float]:
    """Train the autoencoder and return the loss history.

    Parameters
    ----------
    model: Autoencoder
        The network to train.
    data: torch.Tensor
        Input data of shape (N, input_dim).
    epochs: int
        Maximum number of epochs.
    batch_size: int
        Batch size.
    lr: float
        Learning rate.
    weight_decay: float
        L2 regularisation.
    device: torch.device | None
        Device to run on; defaults to GPU if available.
    early_stop_patience: int
        Number of epochs with no improvement before stopping.
    max_grad_norm: Optional[float]
        If set, gradients will be clipped to this norm.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = MSELoss()
    history: List[float] = []

    best_loss = math.inf
    patience = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        # Early stopping
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                break

    return history


__all__ = ["Autoencoder", "AutoencoderConfig", "train_autoencoder"]
