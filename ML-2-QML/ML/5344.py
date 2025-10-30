"""Hybrid convolutional network that merges classical conv, auto‑encoder and a classical sampler or estimator.

This module defines :class:`HybridConvNet`, a PyTorch model that
* applies a 2‑D convolution with a learnable threshold,
* compresses the feature map through a small auto‑encoder,
* and finally classifies or regresses using a classical sampler or estimator
  network.  The architecture is a drop‑in replacement for Conv.py
  while offering the flexibility of a quantum‑aware design.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class HybridConvNet(nn.Module):
    """Hybrid convolutional network combining classical conv, auto‑encoder
    and a classical sampler or estimator.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        latent_dim: int = 16,
        hidden_dims: tuple[int, int] = (64, 32),
        dropout: float = 0.1,
        classifier: str = "sampler",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.threshold = conv_threshold

        # auto‑encoder
        encoder_layers = []
        in_dim = kernel_size * kernel_size
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, kernel_size * kernel_size))
        self.decoder = nn.Sequential(*decoder_layers)

        # classifier
        if classifier == "sampler":
            self.classifier = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        conv_out = self.conv(x)
        act = torch.sigmoid(conv_out - self.threshold)
        flat = act.view(act.size(0), -1)
        latent = self.encoder(flat)
        recon = self.decoder(latent)
        # use the first two decoded values as features for the classifier
        feats = recon[:, :2]
        return self.classifier(feats)

__all__ = ["HybridConvNet"]
