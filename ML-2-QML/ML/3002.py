"""Convolutional autoencoder combining classical conv filter and MLP autoencoder.

The ConvAutoencoder class first applies a 2‑D convolution to the input
patch and then feeds the flattened activations through a fully‑connected
autoencoder that learns a latent representation.  The `run` method
returns the mean‑squared‑error between the input and its reconstruction,
making the class suitable as a drop‑in replacement for the original Conv
filter while adding denoising and dimensionality‑reduction capabilities.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple
from dataclasses import dataclass

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

class ConvAutoencoder(nn.Module):
    """Drop‑in replacement for Conv filter with optional autoencoding."""

    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=kernel_size * kernel_size,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(data)
        activations = torch.sigmoid(conv_out - self.conv_threshold)
        flat = activations.view(activations.size(0), -1)
        recon = self.autoencoder(flat)
        return recon.view_as(conv_out)

    def run(self, data: torch.Tensor | list[list[float]]) -> float:
        """Return the reconstruction MSE for a single sample."""
        self.eval()
        with torch.no_grad():
            tensor = _as_tensor(data)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            recon = self.forward(tensor)
            loss = nn.functional.mse_loss(recon, tensor, reduction="mean")
            return loss.item()

__all__ = ["ConvAutoencoder"]
