"""Enhanced fully‑connected autoencoder with flexible architecture and training utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer
from torch.nn.functional import mse_loss


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
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU
    batch_norm: bool = False
    early_stop_patience: int = 10  # epochs without improvement to stop training


class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with optional batch‑norm and configurable activation."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            if config.batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden))
            encoder_layers.append(config.activation())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            if config.batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden))
            decoder_layers.append(config.activation())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return latent representation of *inputs*."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct *latents* back to input space."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU,
    batch_norm: bool = False,
) -> AutoencoderNet:
    """Factory that returns a configured :class:`AutoencoderNet`."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        batch_norm=batch_norm,
    )
    return AutoencoderNet(config)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    validation_split: float = 0.0,
    early_stop_patience: Optional[int] = None,
) -> list[float]:
    """
    Train *model* on *data* with optional early stopping.
    Returns a history of training loss per epoch.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(_as_tensor(data))
    if validation_split > 0.0:
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        train_ds, val_ds = dataset, None

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer: Optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    history: list[float] = []

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = early_stop_patience or model.config.early_stop_patience

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(train_ds)
        history.append(epoch_loss)

        if val_ds is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (val_batch,) in DataLoader(val_ds, batch_size=batch_size):
                    val_batch = val_batch.to(device)
                    recon = model(val_batch)
                    val_loss += mse_loss(recon, val_batch).item() * val_batch.size(0)
            val_loss /= len(val_ds)
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history


def compute_latent(model: AutoencoderNet, data: torch.Tensor) -> torch.Tensor:
    """Return the latent representation for *data*."""
    model.eval()
    with torch.no_grad():
        return model.encode(_as_tensor(data).to(next(model.parameters()).device))


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
    "compute_latent",
]
