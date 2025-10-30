"""Hybrid ML model combining quanvolution, self‑attention, auto‑encoder and a fully‑connected head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ----- Classical components -----

class QuanvolutionFilter(nn.Module):
    """Simple 2×2 stride‑2 convolution that emulates a quanvolution filter."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class ClassicalSelfAttention(nn.Module):
    """Learnable self‑attention block that mirrors the quantum interface."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, seq_len, embed_dim)
        query = inputs @ self.rotation
        key   = inputs @ self.entangle
        scores = F.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

class AutoencoderConfig:
    """Configuration for the auto‑encoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Tuple[int,...] = (128, 64), dropout: float = 0.1) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class FullyConnectedLayer(nn.Module):
    """Linear head that emulates the quantum fully‑connected layer."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# ----- Hybrid network -----

class QuanvolutionHybridNet(nn.Module):
    """
    End‑to‑end model that chains a quanvolution filter, a self‑attention block,
    an auto‑encoder and a fully‑connected head.
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 10,
                 embed_dim: int = 4,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int,...] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels, out_channels=embed_dim)
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.autoencoder = AutoencoderNet(AutoencoderConfig(
            input_dim=embed_dim * 14 * 14,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout))
        self.fc = FullyConnectedLayer(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quanvolution
        features = self.qfilter(x)  # (batch, 784)
        # 2. Reshape for attention: (batch, seq_len, embed_dim)
        seq_len = features.size(1) // self.attention.embed_dim
        features = features.view(features.size(0), seq_len, self.attention.embed_dim)
        # 3. Self‑attention
        attn = self.attention(features)  # (batch, seq_len, embed_dim)
        # 4. Flatten back to (batch, 784)
        attn = attn.view(attn.size(0), -1)
        # 5. Auto‑encoder
        latent = self.autoencoder.encode(attn)
        # 6. Classification head
        logits = self.fc(latent)
        return F.log_softmax(logits, dim=-1)
