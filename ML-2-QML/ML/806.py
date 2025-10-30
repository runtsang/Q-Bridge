"""Enhanced PyTorch autoencoder with flexible architecture and training utilities.

This module extends the original lightweight autoencoder by adding support for
custom activation functions, optional batch‑norm layers, residual connections,
and a convenient training helper that logs loss history and can plot the
reconstruction error over epochs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


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
    residual: bool = False


class AutoencoderNet(nn.Module):
    """A configurable multilayer perceptron autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = self.config.input_dim
        for hidden in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(self.config.activation())
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
            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(self.config.activation())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.config.input_dim))
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the latent representation of *inputs*."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Return the reconstruction of *latents*."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

    def get_latent(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that returns the latent vector."""
        return self.encode(inputs)


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU,
    batch_norm: bool = False,
    residual: bool = False,
) -> AutoencoderNet:
    """Factory that returns a configured :class:`AutoencoderNet`."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
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
    plot: bool = False,
    early_stop: Optional[int] = None,
) -> list[float]:
    """Train *model* on *data* and return the loss history.

    Parameters
    ----------
    model : AutoencoderNet
        The network to train.
    data : torch.Tensor
        Training data of shape ``(N, input_dim)``.
    epochs : int, default 100
        Number of training epochs.
    batch_size : int, default 64
        Size of mini‑batches.
    lr : float, default 1e-3
        Learning rate for the Adam optimiser.
    weight_decay : float, default 0.0
        L2 regularisation strength.
    device : torch.device | None, default None
        Target device; defaults to CUDA if available.
    plot : bool, default False
        If ``True`` a live plot of the loss curve is shown.
    early_stop : int | None, default None
        Number of epochs with no improvement after which training stops.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    best_loss = float("inf")
    patience = 0

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

        if plot:
            plt.ion()
            plt.clf()
            plt.plot(history, label="train loss")
            plt.xlabel("epoch")
            plt.ylabel("MSE")
            plt.legend()
            plt.pause(0.01)

        if early_stop is not None:
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
            if patience >= early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if plot:
        plt.ioff()
        plt.show()

    return history


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
]
