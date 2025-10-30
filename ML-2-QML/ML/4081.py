"""Pure‑classical hybrid autoencoder with weight clipping.

The network mirrors the original Autoencoder but replaces the standard
linear layers with `ClippedLinear` layers that clamp weights and biases to
keep them bounded, as inspired by the FraudDetection example.  The
architecture is fully compatible with PyTorch and can be used as a
drop‑in replacement for the seed implementation.

Public API
----------
    model = HybridAutoencoder(config)
    reconstruction = model(data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


class ClippedLinear(nn.Module):
    """Linear layer that clips its parameters after each forward pass."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        clip_range: Tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.clip_low, self.clip_high = clip_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp weights and biases in-place.
        with torch.no_grad():
            self.linear.weight.clamp_(self.clip_low, self.clip_high)
            if self.linear.bias is not None:
                self.linear.bias.clamp_(self.clip_low, self.clip_high)
        return self.linear(x)


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    clip_range: Tuple[float, float] = (-5.0, 5.0)


class HybridAutoencoder(nn.Module):
    """Classical hybrid autoencoder with clipped linear layers."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(
                ClippedLinear(in_dim, hidden, clip_range=config.clip_range)
            )
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(
            ClippedLinear(in_dim, config.latent_dim, clip_range=config.clip_range)
        )
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(
                ClippedLinear(in_dim, hidden, clip_range=config.clip_range)
            )
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(
            ClippedLinear(in_dim, config.input_dim, clip_range=config.clip_range)
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass."""
        return self.decode(self.encode(x))


__all__ = ["AutoencoderConfig", "HybridAutoencoder"]
