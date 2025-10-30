"""Autoencoder module with linear and convolutional support, early stopping, and flexible training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence, Callable, Optional, Any

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
    return tensor.to(dtype=torch.float32)


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    input_dim: int | Tuple[int, int, int]  # (channels, H, W) for conv, otherwise flat
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    conv: bool = False
    kernel_size: int = 3
    stride: int = 2
    batch_norm: bool = False
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU
    optimizer_cls: type[Optimizer] = torch.optim.Adam
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stop_patience: int = 10  # epochs with no improvement before stopping


class AutoencoderNet(nn.Module):
    """Flexible autoencoder supporting linear or convolutional layers."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        if config.conv:
            self._build_conv_autoencoder()
        else:
            self._build_linear_autoencoder()

    # ------------------------------------------------------------------
    # Linear architecture
    # ------------------------------------------------------------------
    def _build_linear_autoencoder(self) -> None:
        encoder_layers: list[nn.Module] = []
        in_dim = self.config.input_dim
        for hidden in self.config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(self.config.activation())
            if self.config.batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden))
            if self.config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, self.config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        in_dim = self.config.latent_dim
        for hidden in reversed(self.config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(self.config.activation())
            if self.config.batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden))
            if self.config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, self.config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    # ------------------------------------------------------------------
    # Convolutional architecture
    # ------------------------------------------------------------------
    def _build_conv_autoencoder(self) -> None:
        c, h, w = self.config.input_dim
        encoder_layers: list[nn.Module] = []
        in_channels = c
        for hidden in self.config.hidden_dims:
            encoder_layers.append(
                nn.Conv2d(in_channels, hidden, self.config.kernel_size, stride=self.config.stride, padding=1)
            )
            encoder_layers.append(self.config.activation())
            if self.config.batch_norm:
                encoder_layers.append(nn.BatchNorm2d(hidden))
            if self.config.dropout > 0.0:
                encoder_layers.append(nn.Dropout2d(self.config.dropout))
            in_channels = hidden
        # Flatten to latent vector
        encoder_layers.append(nn.Flatten())
        # Compute flattened size after conv layers
        dummy = torch.zeros(1, *self.config.input_dim)
        with torch.no_grad():
            flat_size = encoder_layers[-2](dummy).shape[1]
        encoder_layers.append(nn.Linear(flat_size, self.config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder mirrors encoder
        decoder_layers: list[nn.Module] = []
        decoder_layers.append(nn.Linear(self.config.latent_dim, flat_size))
        decoder_layers.append(nn.Unflatten(1, (self.config.hidden_dims[-1], h // 2 ** len(self.config.hidden_dims), w // 2 ** len(self.config.hidden_dims))))
        for hidden in reversed(self.config.hidden_dims[:-1]):
            decoder_layers.append(
                nn.ConvTranspose2d(
                    hidden + 1, hidden, self.config.kernel_size, stride=self.config.stride, padding=1
                )
            )
            decoder_layers.append(self.config.activation())
            if self.config.batch_norm:
                decoder_layers.append(nn.BatchNorm2d(hidden))
            if self.config.dropout > 0.0:
                decoder_layers.append(nn.Dropout2d(self.config.dropout))
        decoder_layers.append(
            nn.ConvTranspose2d(
                self.config.hidden_dims[0] + 1, c, self.config.kernel_size, stride=self.config.stride, padding=1
            )
        )
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int | Tuple[int, int, int],
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    conv: bool = False,
    kernel_size: int = 3,
    stride: int = 2,
    batch_norm: bool = False,
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU,
    optimizer_cls: type[Optimizer] = torch.optim.Adam,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    early_stop_patience: int = 10,
) -> AutoencoderNet:
    """Factory that returns a configured :class:`AutoencoderNet`."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        conv=conv,
        kernel_size=kernel_size,
        stride=stride,
        batch_norm=batch_norm,
        activation=activation,
        optimizer_cls=optimizer_cls,
        lr=lr,
        weight_decay=weight_decay,
        early_stop_patience=early_stop_patience,
    )
    return AutoencoderNet(config)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    device: torch.device | None = None,
    early_stop_patience: Optional[int] = None,
) -> list[float]:
    """Train the autoencoder with early stopping and return loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    lr = lr if lr is not None else model.config.lr
    weight_decay = weight_decay if weight_decay is not None else model.config.weight_decay
    optimizer = model.config.optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = mse_loss
    history: list[float] = []
    best_loss = float("inf")
    patience = early_stop_patience if early_stop_patience is not None else model.config.early_stop_patience
    counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    return history


__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
