"""PyTorch implementation of an advanced autoencoder with skip‑connections, batch‑norm and latent regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

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
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    batch_norm: bool = False
    skip_connections: bool = False
    latent_reg: float = 0.0  # weight for latent regularization term


class AutoencoderNet(nn.Module):
    """A flexible multi‑layer perceptron autoencoder with optional skip‑connections and batch‑norm."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        if self.config.skip_connections:
            # Simple residual connection from input to output
            self.residual = nn.Linear(self.config.input_dim, self.config.input_dim)

    def _build_encoder(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = self.config.input_dim
        for hidden in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.config.latent_dim))
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = self.config.latent_dim
        for hidden in reversed(self.config.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.config.input_dim))
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        decoded = self.decode(latent)
        if self.config.skip_connections:
            decoded = decoded + self.residual(inputs)
        return decoded


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    batch_norm: bool = False,
    skip_connections: bool = False,
    latent_reg: float = 0.0
) -> AutoencoderNet:
    """Factory that returns a configured :class:`AutoencoderNet`."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        batch_norm=batch_norm,
        skip_connections=skip_connections,
        latent_reg=latent_reg,
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
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            latent = model.encode(batch)
            reconstruction = model.decode(latent)
            loss = loss_fn(reconstruction, batch)
            if model.config.latent_reg > 0.0:
                loss += model.config.latent_reg * torch.mean(latent**2)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
]
