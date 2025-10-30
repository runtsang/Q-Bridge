"""Hybrid classical self‑attention leveraging a PyTorch autoencoder."""
from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class HybridConfig:
    embed_dim: int
    latent_dim: int = 16
    hidden_dims: Tuple[int, int] = (64, 32)

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        enc_layers = []
        in_dim = cfg.embed_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.embed_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class HybridSelfAttention(nn.Module):
    """Classical hybrid attention: autoencoder → attention."""
    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.auto = AutoencoderNet(cfg)
        self.scale = 1.0 / (cfg.latent_dim ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Autoencoder feature extraction
        z = self.auto.encode(x)
        # Self‑attention on latent space
        q = torch.matmul(z, z.t())
        scores = torch.softmax(q * self.scale, dim=-1)
        return torch.matmul(scores, z)

def HybridSelfAttentionFactory(embed_dim: int) -> HybridSelfAttention:
    cfg = HybridConfig(embed_dim=embed_dim)
    return HybridSelfAttention(cfg)

__all__ = ["HybridSelfAttention", "HybridSelfAttentionFactory"]
