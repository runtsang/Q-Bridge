"""Hybrid classical model that fuses convolution, quantum‑like filtering, and an autoencoder.

This module defines :class:`ConvAutoencoderFusion`, a fully‑differentiable PyTorch
model that first applies a learnable convolutional filter with a threshold,
then encodes the flattened features into a latent space, and finally decodes
them back to the original dimension.  The architecture mirrors the quantum
counterpart in the QML module, enabling side‑by‑side experiments.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class ConvFilter(nn.Module):
    """2‑D convolutional filter with learnable threshold and optional dropout."""
    def __init__(self, kernel_size: int = 2, in_channels: int = 1,
                 out_channels: int = 1, threshold: float = 0.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(dropout)
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = F.sigmoid(out - self.threshold)
        return out.mean(dim=(2, 3)).unsqueeze(1)  # shape (batch, 1)

class Encoder(nn.Module):
    """Fully‑connected encoder mirroring the classical Autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Decoder(nn.Module):
    """Fully‑connected decoder mirroring the classical Autoencoder."""
    def __init__(self, latent_dim: int, output_dim: int,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ConvAutoencoderFusion(nn.Module):
    """Hybrid model that stacks a convolutional filter, an encoder and a decoder."""
    def __init__(self,
                 in_channels: int = 1,
                 kernel_size: int = 2,
                 latent_dim: int = 3,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=kernel_size,
                               in_channels=in_channels,
                               out_channels=1,
                               threshold=0.0,
                               dropout=dropout)
        # The flattened feature dimension after conv filter
        self.encoder = Encoder(input_dim=1, latent_dim=latent_dim,
                               hidden_dims=hidden_dims, dropout=dropout)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=1,
                               hidden_dims=hidden_dims, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, H, W)
        conv_out = self.conv(x)          # (batch, 1)
        encoded = self.encoder(conv_out)
        decoded = self.decoder(encoded)
        return decoded

    def run(self, data) -> torch.Tensor:
        """Convenience method to process numpy or torch input."""
        if isinstance(data, torch.Tensor):
            tensor = data
        else:
            tensor = torch.as_tensor(data, dtype=torch.float32)
        if tensor.ndim == 3:  # (H, W, C) or (H, W)
            tensor = tensor.unsqueeze(0)  # add batch
        if tensor.ndim == 4 and tensor.shape[1]!= 1:
            # assume shape (batch, H, W, C)
            tensor = tensor.permute(0, 3, 1, 2)
        return self.forward(tensor)

__all__ = ["ConvAutoencoderFusion"]
