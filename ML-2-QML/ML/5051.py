"""
Hybrid classical autoencoder with quantum‑informed latent space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderGen380Config:
    """Hyper‑parameters for :class:`AutoencoderGen380Net`."""
    input_dim: int
    latent_dim: int = 16
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    attention_dim: int = 4

# --------------------------------------------------------------------------- #
# Classical Self‑Attention Utility
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block that mirrors the quantum version."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q @ k.transpose(-1, -2) / (self.embed_dim ** 0.5), dim=-1)
        return scores @ v

# --------------------------------------------------------------------------- #
# Classical Autoencoder
# --------------------------------------------------------------------------- #
class AutoencoderGen380Net(nn.Module):
    """Hybrid autoencoder that mixes classical MLP, self‑attention and a quantum encoder."""

    def __init__(self, cfg: AutoencoderGen380Config) -> None:
        super().__init__()
        self.cfg = cfg

        # Encoder
        enc_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            enc_layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Self‑attention applied to latent space
        self.attention = ClassicalSelfAttention(cfg.attention_dim)

        # Decoder
        dec_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            dec_layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Quantum module placeholder – will be attached externally
        self.quantum_latent = None

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical encoding
        z = self.encoder(x)

        # Augment with quantum latent if available
        if self.quantum_latent is not None:
            z = z + self.quantum_latent(x)

        # Attention on enriched latent
        z = self.attention(z)

        # Classical decoding
        return self.decoder(z)

    # --------------------------------------------------------------------- #
    # Attach quantum latent extractor
    # --------------------------------------------------------------------- #
    def attach_quantum(self, q_module: nn.Module) -> None:
        """Attach a quantum module that returns a vector of shape (batch, latent_dim)."""
        self.quantum_latent = q_module

# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: AutoencoderGen380Net,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(TensorDataset(_as_tensor(data)), batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
# Helper to cast data to float32 tensor
# --------------------------------------------------------------------------- #
def _as_tensor(data: torch.Tensor | list | tuple | list[float] | tuple[float,...]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.float()
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = [
    "AutoencoderGen380Net",
    "AutoencoderGen380Config",
    "train_autoencoder",
    "ClassicalSelfAttention",
]
