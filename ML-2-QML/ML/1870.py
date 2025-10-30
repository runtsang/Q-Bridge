"""Hybrid autoencoder with optional residual connections and batch normalization."""

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
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    batch_norm: bool = False
    residual: bool = False


class AutoencoderNet(nn.Module):
    """A configurable multilayer perceptron autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._build_mlp(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.latent_dim,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            residual=config.residual,
        )
        self.decoder = self._build_mlp(
            input_dim=config.latent_dim,
            hidden_dims=list(reversed(config.hidden_dims)),
            output_dim=config.input_dim,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            residual=config.residual,
        )

    def _build_mlp(
        self,
        input_dim: int,
        hidden_dims: Tuple[int,...],
        output_dim: int,
        dropout: float,
        batch_norm: bool,
        residual: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden in hidden_dims:
            linear = nn.Linear(in_dim, hidden)
            layers.append(linear)
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            # Add residual skip when dimensions match
            if residual and hidden == in_dim:
                layers.append(nn.Identity())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    batch_norm: bool = False,
    residual: bool = False,
) -> AutoencoderNet:
    """Factory that returns a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        batch_norm=batch_norm,
        residual=residual,
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
    early_stop: bool = False,
    patience: int = 10,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    best_loss = float("inf")
    no_improve = 0

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

        # Early stopping logic
        if early_stop:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return history


__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
