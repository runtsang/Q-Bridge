"""Hybrid classical autoencoder with an integrated self‑attention block.

The design merges the lightweight MLP autoencoder from the original seed
with a self‑attention mechanism inspired by the second reference.
Training is performed with a standard Adam optimiser and the loss history
is exposed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# Configuration
# ----------------------------
@dataclass
class AutoencoderHybridConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    attention_heads: int = 4  # number of parallel attention heads

# ----------------------------
# Self‑Attention helper
# ----------------------------
class SimpleSelfAttention(nn.Module):
    """A lightweight multi‑head attention layer operating on the feature tensor."""
    def __init__(self, embed_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5

        # linear projections for queries, keys and values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # split into heads
        B, N = Q.shape
        head_dim = self.embed_dim // self.heads
        Q = Q.view(B, self.heads, head_dim)
        K = K.view(B, self.heads, head_dim)
        V = V.view(B, self.heads, head_dim)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = torch.softmax(scores, dim=-1)

        out = torch.bmm(attn, V)
        out = out.reshape(B, self.embed_dim)
        return self.out_proj(out)

# ----------------------------
# Autoencoder
# ----------------------------
class AutoencoderHybrid(nn.Module):
    """Multi‑layer perceptron autoencoder with an embedded attention block."""
    def __init__(self, cfg: AutoencoderHybridConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden

        # attention sub‑module
        encoder_layers.append(SimpleSelfAttention(in_dim, cfg.attention_heads))

        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden

        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.encode(x)
        return self.decode(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

# ----------------------------
# Factory
# ----------------------------
def AutoencoderHybrid(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    attention_heads: int = 4,
) -> AutoencoderHybrid:
    cfg = AutoencoderHybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        attention_heads=attention_heads,
    )
    return AutoencoderHybrid(cfg)

# ----------------------------
# Training helper
# ----------------------------
def train_autoencoder(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Standard reconstruction training loop."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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

# ----------------------------
# Utility
# ----------------------------
def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = [
    "AutoencoderHybrid",
    "AutoencoderHybridConfig",
    "train_autoencoder",
]
