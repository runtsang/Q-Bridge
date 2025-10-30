"""Enhanced PyTorch autoencoder with denoising, skip connections, and latent regularization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current device, preserving dtype."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    noise_std: float = 0.0          # standard deviation of Gaussian noise added to input
    latent_reg_weight: float = 0.0  # weight of L2 penalty on latent vector


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder with denoising and skip connections."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_dims = config.hidden_dims

        # Encoder
        self.encoder_layers: list[tuple[nn.Module, nn.Module, nn.Module]] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            self.encoder_layers.append(
                (nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(config.dropout))
            )
            in_dim = hidden
        self.encoder_last = nn.Linear(in_dim, config.latent_dim)

        # Decoder
        self.decoder_layers: list[tuple[nn.Module, nn.Module, nn.Module]] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            self.decoder_layers.append(
                (nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(config.dropout))
            )
            in_dim = hidden
        self.decoder_last = nn.Linear(in_dim, config.input_dim)

    # ----------------------------------------------------------------------- #
    # Forward passes
    # ----------------------------------------------------------------------- #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input into a latent representation."""
        h = x
        for linear, relu, dropout in self.encoder_layers:
            residual = h
            h = linear(h)
            h = relu(h)
            h = dropout(h)
            if residual.shape[-1] == h.shape[-1]:
                h = h + residual
        latent = self.encoder_last(h)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector back to the input space."""
        h = z
        for linear, relu, dropout in self.decoder_layers:
            residual = h
            h = linear(h)
            h = relu(h)
            h = dropout(h)
            if residual.shape[-1] == h.shape[-1]:
                h = h + residual
        recon = self.decoder_last(h)
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Full autoencoder forward pass with optional denoising."""
        if self.config.noise_std > 0.0:
            noise = torch.randn_like(x) * self.config.noise_std
            x = x + noise
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    noise_std: float = 0.0,
    latent_reg_weight: float = 0.0,
) -> AutoencoderNet:
    """Instantiate a configured :class:`AutoencoderNet`."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        noise_std=noise_std,
        latent_reg_weight=latent_reg_weight,
    )
    return AutoencoderNet(config)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
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
    """
    Simple reconstruction training loop.

    Parameters
    ----------
    model : AutoencoderNet
        The autoencoder to train.
    data : torch.Tensor
        Input data of shape (N, input_dim).
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Batch size.
    lr : float, optional
        Learning rate.
    weight_decay : float, optional
        Weight decay for Adam.
    device : torch.device | None, optional
        Device to run on. Defaults to CUDA if available.

    Returns
    -------
    history : list[float]
        Mean loss per epoch.
    """
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
            recon = model(batch)
            mse = loss_fn(recon, batch)
            # Latent regularization
            latent = model.encode(batch)
            latent_reg = model.config.latent_reg_weight * torch.mean(latent**2)
            loss = mse + latent_reg
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
