"""Hybrid Autoencoder with integrated self‑attention.

This module combines the simple fully‑connected autoencoder from
`Autoencoder.py` with a classical self‑attention block inspired by
`SelfAttention.py`.  The attention layer is applied after each hidden
layer in the encoder and before each hidden layer in the decoder,
allowing the network to learn context‑aware latent representations.
"""

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, Iterable

# Classical self‑attention layer ------------------------------------------------
class SelfAttentionLayer(nn.Module):
    """A lightweight self‑attention block that operates on a batch of
    feature vectors.  It mirrors the behaviour of the reference
    implementation but is fully differentiable and can be trained
    end‑to‑end with the autoencoder.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable parameters that emulate the rotation and entangle
        # matrices in the reference code.
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        query = x @ self.rotation
        key   = x @ self.entangle
        scores = torch.softmax(query @ key.transpose(-1, -2) /
                               (self.embed_dim ** 0.5), dim=-1)
        return scores @ x

# Autoencoder configuration ----------------------------------------------------
@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

# Main model ---------------------------------------------------------------
class HybridAutoencoder(nn.Module):
    """Fully‑connected autoencoder that inserts a self‑attention layer
    after every hidden block.  The attention layer learns to
    re‑weight features before they are passed to the next linear
    transformation, improving the expressivity of the latent space.
    """
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.encoder = nn.ModuleList()
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            self.encoder.append(nn.Linear(in_dim, hidden))
            self.encoder.append(nn.ReLU())
            self.encoder.append(SelfAttentionLayer(hidden))
            if config.dropout > 0.0:
                self.encoder.append(nn.Dropout(config.dropout))
            in_dim = hidden
        self.encoder.append(nn.Linear(in_dim, config.latent_dim))

        self.decoder = nn.ModuleList()
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            self.decoder.append(nn.Linear(in_dim, hidden))
            self.decoder.append(nn.ReLU())
            self.decoder.append(SelfAttentionLayer(hidden))
            if config.dropout > 0.0:
                self.decoder.append(nn.Dropout(config.dropout))
            in_dim = hidden
        self.decoder.append(nn.Linear(in_dim, config.input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            z = layer(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.decode(self.encode(x))

# Factory ---------------------------------------------------------------
def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoder:
    """Convenience constructor mirroring the original API."""
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoder(cfg)

# Training loop -----------------------------------------------------------
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
    """Standard reconstruction training loop."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# Helper for tensor conversion ---------------------------------------------
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
    "HybridAutoencoderConfig",
    "SelfAttentionLayer",
    "train_hybrid_autoencoder",
]
