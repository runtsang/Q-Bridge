"""
PyTorch implementation of a hybrid variational autoencoder with optional residual connections.
"""

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
    """Configuration values for :class:`AutoencoderHybrid`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    residual: bool = False
    vae: bool = False  # if True, outputs mean & logvar for reparameterization


class AutoencoderHybrid(nn.Module):
    """A versatile autoencoder that supports residual connections and VAE behaviour."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.vae = config.vae

        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim * (2 if self.vae else 1)))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim * (2 if self.vae else 1)
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        if config.residual:
            # Simple skip connection from input to reconstruction (identity)
            self.residual = nn.Identity()
        else:
            self.residual = None

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs into latent representation."""
        latent = self.encoder(inputs)
        if self.vae:
            # Split into mean & logvar
            mean, logvar = latent.chunk(2, dim=-1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return latent

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents back to data space."""
        recon = self.decoder(latents)
        if self.residual is not None:
            recon += self.residual(recon)  # simple residual
        return recon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Full autoencoder forward pass."""
        return self.decode(self.encode(inputs))


def AutoencoderHybridFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    residual: bool = False,
    vae: bool = False,
) -> AutoencoderHybrid:
    """Convenience factory mirroring the original seed."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        residual=residual,
        vae=vae,
    )
    return AutoencoderHybrid(config)


def train_autoencoder(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train loop that records MSE loss history."""
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
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["AutoencoderHybrid", "AutoencoderHybridFactory", "AutoencoderConfig", "train_autoencoder"]
