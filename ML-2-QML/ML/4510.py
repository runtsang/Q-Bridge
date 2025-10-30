"""Hybrid estimator that combines autoencoding, self‑attention, convolutional feature extraction, and a classical regression head.

The architecture is built from the four reference seeds, but all components are re‑wired to form a single end‑to‑end model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple


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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


# ---------- Self‑Attention ----------
class ClassicalSelfAttention:
    """Simple self‑attention block that mirrors the quantum interface."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


# ---------- Convolutional feature extractor ----------
class QFCModel(nn.Module):
    """Convolutional feature extractor adapted for 1×1 inputs."""

    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 1 * 1, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.features(x)
        flattened = features.view(x.shape[0], -1)
        out = self.fc(flattened)
        return self.norm(out)


# ---------- Hybrid Estimator ----------
class HybridEstimatorQNN(nn.Module):
    """
    End‑to‑end estimator that stitches together:

    1. An autoencoder for dimensionality reduction.
    2. A self‑attention module with trainable rotation/entangle parameters.
    3. A lightweight convolutional extractor (QFCModel).
    4. A small regression head.

    The design preserves the modularity of the original seeds while enabling richer feature learning.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        attention_dim: int = 4,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(input_dim, latent_dim=latent_dim)
        self.attention = ClassicalSelfAttention(attention_dim)
        # Trainable parameters for the attention block
        self.rotation_params = nn.Parameter(torch.randn(attention_dim, attention_dim))
        self.entangle_params = nn.Parameter(torch.randn(attention_dim, attention_dim))
        self.conv_extractor = QFCModel(attention_dim)
        self.regressor = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # 1. Autoencoder encoding
        latent = self.autoencoder.encode(x)  # shape (B, latent_dim)

        # 2. Self‑attention
        attn_out = self.attention.run(
            self.rotation_params.detach().cpu().numpy(),
            self.entangle_params.detach().cpu().numpy(),
            latent.detach().cpu().numpy(),
        )
        attn_tensor = torch.from_numpy(attn_out).float().to(x.device)

        # 3. Convolutional feature extraction
        conv_input = attn_tensor.unsqueeze(1).unsqueeze(2)  # (B,1,1,attention_dim)
        conv_features = self.conv_extractor(conv_input)  # (B,4)

        # 4. Regression head
        out = self.regressor(conv_features)  # (B, output_dim)
        return out.squeeze(-1)
