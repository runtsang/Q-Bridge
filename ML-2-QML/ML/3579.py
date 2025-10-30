"""Hybrid classical auto‑encoder with integrated self‑attention.

This module builds on the original fully‑connected Autoencoder by adding a
self‑attention mechanism that operates on the latent space.  The design
mirrors the quantum interface so that the same factory function can be
used in both the classical and quantum settings, facilitating side‑by‑side
experiments.

Typical usage::
    >>> from Autoencoder__gen207 import Autoencoder
    >>> model = Autoencoder(input_dim=784, latent_dim=32)
    >>> loss_history = train_autoencoder_gen207(model, data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: torch.Tensor | List[float]) -> torch.Tensor:
    """Ensure input is a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderGen207Config:
    """Configuration for :class:`AutoencoderGen207`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    attention_heads: int = 4
    attention_dim: int = 16


class SelfAttention(nn.Module):
    """A lightweight multi‑head self‑attention block."""

    def __init__(self, embed_dim: int, heads: int = 4, head_dim: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, heads * head_dim * 3, bias=False)
        self.out_proj = nn.Linear(heads * head_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embed = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, self.heads, 3 * self.head_dim)
        q, k, v = qkv[..., :self.head_dim], qkv[..., self.head_dim:2*self.head_dim], qkv[..., 2*self.head_dim:]
        scores = torch.einsum("bshd,bshd->bshh", q, k) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bshh,bshd->bshd", attn, v)
        out = out.reshape(batch, seq_len, self.heads * self.head_dim)
        return self.out_proj(out)


class AutoencoderGen207(nn.Module):
    """Classical auto‑encoder that processes the latent representation
    with a self‑attention block before decoding."""

    def __init__(self, config: AutoencoderGen207Config) -> None:
        super().__init__()
        # Encoder
        encoder_layers: List[nn.Module] = []
        inp = config.input_dim
        for h in config.hidden_dims:
            encoder_layers.append(nn.Linear(inp, h))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            inp = h
        encoder_layers.append(nn.Linear(inp, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Attention on latent
        self.attention = SelfAttention(
            embed_dim=config.latent_dim,
            heads=config.attention_heads,
            head_dim=config.attention_dim,
        )

        # Decoder
        decoder_layers: List[nn.Module] = []
        inp = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(inp, h))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            inp = h
        decoder_layers.append(nn.Linear(inp, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        # Expand to a sequence dimension for attention
        z_seq = z.unsqueeze(1)  # (batch, seq=1, dim)
        z_att = self.attention(z_seq).squeeze(1)
        return self.decode(z_att)


def Autoencoder(*, input_dim: int, latent_dim: int = 32,
                hidden_dims: Tuple[int,...] = (128, 64),
                dropout: float = 0.1, attention_heads: int = 4,
                attention_dim: int = 16) -> AutoencoderGen207:
    """Factory producing a configured :class:`AutoencoderGen207`."""
    cfg = AutoencoderGen207Config(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        attention_heads=attention_heads,
        attention_dim=attention_dim,
    )
    return AutoencoderGen207(cfg)


def train_autoencoder_gen207(
    model: AutoencoderGen207,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Train ``model`` on ``data`` and return the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "AutoencoderGen207",
    "Autoencoder",
    "AutoencoderGen207Config",
    "train_autoencoder_gen207",
]
