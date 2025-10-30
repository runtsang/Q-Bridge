"""Hybrid quanvolution model combining classical convolution, autoencoder and kernel-based classification.

The module defines a single class QuanvolutionHybrid that can be used as a drop‑in
replacement for the original Quanvolution example.  It extends the classical
convolutional filter with a lightweight fully‑connected auto‑encoder and a
kernel‑based linear classifier.  The design mirrors the quantum version
seamlessly while staying within the PyTorch ecosystem.

Classes
-------
AutoencoderConfig
AutoencoderNet
Kernel
QuanvolutionHybrid
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Sequence

# ---------- Autoencoder ----------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder.append(nn.Linear(in_dim, hidden))
            encoder.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder.append(nn.Linear(in_dim, hidden))
            decoder.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# ---------- Kernel ----------
class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> torch.Tensor:
    kernel = Kernel(gamma)
    return torch.tensor([[kernel(x, y).item() for y in b] for x in a])

# ---------- Hybrid Model ----------
class QuanvolutionHybrid(nn.Module):
    """
    Classical hybrid model that mimics the quantum quanvolution pipeline.
    Architecture:
      * 2×2 convolutional filter
      * Flattened feature vector
      * Classical auto‑encoder (latent_dim=32)
      * Kernel‑based linear classifier using RBF kernel and learnable support vectors
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_filters: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        gamma: float = 1.0,
        num_support_vectors: int = 20,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, num_filters, kernel_size=kernel_size, stride=stride)
        conv_out_dim = num_filters * (28 // stride) * (28 // stride)
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=conv_out_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )
        self.support_vectors = nn.Parameter(torch.randn(num_support_vectors, latent_dim))
        self.kernel = Kernel(gamma)
        self.classifier = nn.Linear(num_support_vectors, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        z = self.autoencoder.encode(x)
        # compute kernel similarity with support vectors
        k = torch.exp(
            -self.kernel.gamma
            * torch.sum((z.unsqueeze(1) - self.support_vectors.unsqueeze(0)) ** 2, dim=2)
        )
        logits = self.classifier(k)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
