"""Hybrid classical module combining self‑attention and auto‑encoding.

The :class:`SelfAttentionHybrid` class bundles a PyTorch multi‑head
scaled‑dot‑product attention block with a lightweight fully‑connected
auto‑encoder.  The two sub‑modules can be chained to form an encoder‑decoder
pipeline, offering a straightforward way to compare classical attention
representations with latent spaces learned by an auto‑encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AttentionConfig:
    """Configuration for the self‑attention block."""
    embed_dim: int
    num_heads: int = 1
    dropout: float = 0.0


class ClassicalSelfAttention(nn.Module):
    """PyTorch implementation of multi‑head scaled‑dot‑product attention."""
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        if cfg.embed_dim % cfg.num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.cfg = cfg
        self.head_dim = cfg.embed_dim // cfg.num_heads
        self.qkv = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.cfg.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B,N,H,D/H)
        scores = torch.einsum("bnhd,bmhd->bhnm", q, k) / (self.head_dim**0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhnm,bmhd->bnhd", attn, v)
        out = out.reshape(B, N, D)
        return self.out_proj(out)


@dataclass
class AutoencoderConfig:
    """Configuration for the fully‑connected auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """A lightweight fully‑connected auto‑encoder."""
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(cfg.dropout)])
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [nn.Linear(cfg.latent_dim, cfg.hidden_dims[-1])]
        for hidden in reversed(cfg.hidden_dims[:-1]):
            decoder_layers.extend([nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(hidden, hidden)])
        decoder_layers.extend([nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dims[0], cfg.input_dim)])
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


class SelfAttentionHybrid(nn.Module):
    """Hybrid encoder‑decoder that chains self‑attention with an auto‑encoder.

    Parameters
    ----------
    attention_cfg : AttentionConfig
        Configuration for the self‑attention block.
    auto_cfg : AutoencoderConfig
        Configuration for the auto‑encoder.
    """
    def __init__(self, attention_cfg: AttentionConfig, auto_cfg: AutoencoderConfig):
        super().__init__()
        self.attn = ClassicalSelfAttention(attention_cfg)
        self.auto = AutoencoderNet(auto_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Apply self‑attention followed by the auto‑encoder."""
        return self.auto(self.attn(x))

    def encode(self, x: Tensor) -> Tensor:
        """Encode input using attention + auto‑encoder."""
        return self.auto.encode(self.attn(x))

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vector using auto‑encoder only."""
        return self.auto.decode(z)


__all__ = ["SelfAttentionHybrid", "AttentionConfig", "AutoencoderConfig", "AutoencoderNet"]
