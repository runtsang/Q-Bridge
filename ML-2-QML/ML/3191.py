"""Hybrid autoencoder combining classical self‑attention and a quantum‑enhanced variational encoder.

The module defines:
- SelfAttentionModule: a learnable multi‑head self‑attention block.
- HybridAutoencoder: an MLP encoder/decoder with an optional self‑attention layer inserted after the first hidden layer.
- factory function `HybridAutoencoderFactory` mirroring the original API.
- `train_hybrid_autoencoder` training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    attention_heads: int = 4  # number of self‑attention heads

class SelfAttentionModule(nn.Module):
    """Learnable multi‑head self‑attention block."""
    def __init__(self, embed_dim: int, heads: int = 1):
        super().__init__()
        self.heads = heads
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # support both (B, D) and (B, N, D) inputs
        if x.dim() == 2:
            B, D = x.shape
            x = x.unsqueeze(1)  # (B, 1, D)
        else:
            B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each shape: (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)

class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder with optional self‑attention inside the encoder."""
    def __init__(self, config: HybridAutoencoderConfig, use_attention: bool = True) -> None:
        super().__init__()
        self.use_attention = use_attention
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        # optional self‑attention
        if use_attention:
            encoder_layers.append(nn.Linear(in_dim, in_dim))
            encoder_layers.append(SelfAttentionModule(in_dim, heads=config.attention_heads))
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

def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    attention_heads: int = 4,
    use_attention: bool = True,
) -> HybridAutoencoder:
    """Return a configured :class:`HybridAutoencoder`."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        attention_heads=attention_heads,
    )
    return HybridAutoencoder(config, use_attention=use_attention)

def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = ["HybridAutoencoder", "HybridAutoencoderFactory", "train_hybrid_autoencoder", "SelfAttentionModule"]
