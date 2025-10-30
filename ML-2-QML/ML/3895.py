"""Hybrid convolutional filter with autoencoder‑style bottleneck for classical training."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Iterable, Tuple

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class AutoencoderNet(nn.Module):
    """Small MLP autoencoder used as a bottleneck."""
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Tuple[int, int], dropout: float = 0.1):
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class ConvGen206(nn.Module):
    """Classical convolutional filter with a latent autoencoder bottleneck."""
    def __init__(
        *,
        kernel_size: int = 2,
        threshold: float = 0.0,
        latent_dim: int = 16,
        hidden_dims: Tuple[int, int] = (64, 64),
        dropout: float = 0.1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        # autoencoder bottleneck operates on flattened output
        self.autoencoder = AutoencoderNet(
            input_dim=kernel_size * kernel_size,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.to(self.device)

    def _forward_conv(self, data: torch.Tensor) -> torch.Tensor:
        """Apply convolution and sigmoid activation."""
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif data.ndim == 3:
            data = data.unsqueeze(0)  # (1,C,H,W)
        conv_out = self.conv(data.to(self.device))
        act = torch.sigmoid(conv_out - self.threshold)
        return act.squeeze(0)  # (1, H', W')

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass returning reconstructed activation map."""
        act = self._forward_conv(data)
        flat = act.view(-1, self.kernel_size * self.kernel_size)
        recon = self.autoencoder(flat)
        recon_map = recon.view(1, 1, self.kernel_size, self.kernel_size)
        return recon_map

    def run(self, data: Iterable[float] | torch.Tensor) -> float:
        """Convenience wrapper that accepts raw data and returns mean activation."""
        tensor = _as_tensor(data).float()
        with torch.no_grad():
            recon_map = self.forward(tensor)
            return recon_map.mean().item()

def ConvGen206Factory(
    kernel_size: int = 2,
    threshold: float = 0.0,
    latent_dim: int = 16,
    hidden_dims: Tuple[int, int] = (64, 64),
    dropout: float = 0.1,
    device: torch.device | None = None,
) -> ConvGen206:
    """Factory mirroring Conv() that returns a configured ConvGen206."""
    return ConvGen206(
        kernel_size=kernel_size,
        threshold=threshold,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        device=device,
    )

__all__ = ["ConvGen206", "ConvGen206Factory", "AutoencoderNet"]
