"""Classical autoencoder + self‑attention + convolutional network for binary classification.

This module defines a purely classical neural network that integrates an autoencoder
for feature compression, a self‑attention block for adaptive weighting,
and a dense head for binary decision making.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AEConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AEConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, embed_dim)
        q = self.query(x).unsqueeze(2)  # (batch, embed_dim, 1)
        k = self.key(x).unsqueeze(1)    # (batch, 1, embed_dim)
        v = self.value(x).unsqueeze(2)  # (batch, embed_dim, 1)
        scores = F.softmax(torch.bmm(q, k) / np.sqrt(self.embed_dim), dim=-1)
        out = torch.bmm(scores, v).squeeze(2)  # (batch, embed_dim)
        return out


class HybridAutoEncoderAttentionNet(nn.Module):
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Compute flattened feature size via dummy input
        dummy = torch.zeros(1, 3, 32, 32)
        flat = self._forward_conv(dummy)
        flat_size = flat.shape[1]

        # Autoencoder on flattened features
        ae_cfg = AEConfig(
            input_dim=flat_size,
            latent_dim=32,
            hidden_dims=(128, 64),
            dropout=0.1,
        )
        self.autoencoder = AutoencoderNet(ae_cfg)

        # Self‑attention on latent vector
        self.attention = SelfAttentionLayer(embed_dim=ae_cfg.latent_dim)

        # Dense head
        self.head = nn.Linear(ae_cfg.latent_dim, 1)
        self.shift = shift

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        latent = self.autoencoder.encode(x)
        attn_out = self.attention(latent)
        logits = self.head(attn_out)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["AEConfig", "AutoencoderNet", "SelfAttentionLayer", "HybridAutoEncoderAttentionNet"]
