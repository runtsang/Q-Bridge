"""Hybrid auto‑encoder combining dense layers, self‑attention, and optional quantum kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
#  Utility helper
# --------------------------------------------------------------------------- #

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

# --------------------------------------------------------------------------- #
#  Classical building blocks
# --------------------------------------------------------------------------- #

@dataclass
class HybridAutoencoderConfig:
    """Configuration for the hybrid encoder/decoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # attention parameters
    use_attention: bool = False
    attn_heads: int = 4
    attn_dim: int = 64

class SelfAttentionBlock(nn.Module):
    """A lightweight multi‑head attention module used inside the encoder."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # query, key, value are the same tensor in self‑attention
        attn_out, _ = self.attn(x, x, x)
        return attn_out

class QuantumKernel:
    """Base class for a kernel that can be evaluated between two tensors."""
    def __init__(self, device: str = "cpu"):
        self.device = device

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a scalar kernel value between two tensors."""
        raise NotImplementedError

class ClassicalRBFKernel(QuantumKernel):
    """Classical RBF kernel matching the interface of the quantum kernel."""
    def __init__(self, gamma: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --------------------------------------------------------------------------- #
#  Encoder / Decoder
# --------------------------------------------------------------------------- #

class HybridEncoder(nn.Module):
    """Encoder that optionally inserts a self‑attention block before the latent layer."""
    def __init__(self, config: HybridAutoencoderConfig):
        super().__init__()
        self.config = config
        layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        self.attn = None
        if config.use_attention:
            self.attn = SelfAttentionBlock(config.attn_dim, config.attn_heads, config.dropout)
        layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn is not None:
            seq_len = x.size(1) // self.config.attn_dim
            assert seq_len * self.config.attn_dim == x.size(1), "Input dimension must be divisible by attn_dim"
            x = x.view(x.size(0), seq_len, self.config.attn_dim)
            x = self.attn(x)
            x = x.reshape(x.size(0), -1)
        return self.encoder(x)

class HybridDecoder(nn.Module):
    """Decoder mirroring the encoder architecture."""
    def __init__(self, config: HybridAutoencoderConfig):
        super().__init__()
        layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class HybridAutoencoder(nn.Module):
    """Full auto‑encoder consisting of an encoder and a decoder."""
    def __init__(self, config: HybridAutoencoderConfig):
        super().__init__()
        self.encoder = HybridEncoder(config)
        self.decoder = HybridDecoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

# --------------------------------------------------------------------------- #
#  Factory and training utilities
# --------------------------------------------------------------------------- #

def hybrid_autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_attention: bool = False,
    attn_heads: int = 4,
    attn_dim: int = 64,
) -> HybridAutoencoder:
    """Convenience factory mirroring the original Autoencoder factory."""
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_attention=use_attention,
        attn_heads=attn_heads,
        attn_dim=attn_dim,
    )
    return HybridAutoencoder(cfg)

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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "hybrid_autoencoder",
    "train_hybrid_autoencoder",
    "SelfAttentionBlock",
    "ClassicalRBFKernel",
]
