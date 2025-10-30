"""Hybrid classical autoencoder combining transformer encoder, self‑attention and kernel‑aware latent space."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    num_heads: int = 4
    ffn_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)


class HybridAutoencoderNet(nn.Module):
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            *[TransformerEncoderLayer(config.input_dim, config.num_heads, config.ffn_dim, config.dropout)
              for _ in range(config.num_layers)]
        )
        self.self_attn = SelfAttentionLayer(config.input_dim)
        self.latent_proj = nn.Linear(config.input_dim, config.latent_dim)
        self.decoder = nn.Sequential(
            *[nn.Linear(config.latent_dim, config.input_dim) for _ in range(config.num_layers)]
        )
        self.output_proj = nn.Linear(config.input_dim, config.input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.self_attn(x)
        z = self.latent_proj(x.mean(dim=1))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = z.unsqueeze(1).repeat(1, self.config.input_dim, 1)
        x = self.decoder(x)
        out = self.output_proj(x.mean(dim=1))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    num_heads: int = 4,
    ffn_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> HybridAutoencoderNet:
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    return HybridAutoencoderNet(cfg)


def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


# Alias for unified API
HybridAutoencoder = HybridAutoencoderNet

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderNet",
    "train_hybrid_autoencoder",
    "HybridAutoencoderConfig",
]
