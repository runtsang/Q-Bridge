"""Hybrid sampler autoencoder combining classical encoder‑decoder with a quantum sampler."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Iterable, Optional

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class HybridConfig:
    """Parameters shared by the classical and quantum parts."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    num_qubits: int = 4  # quantum part dimensionality
    sampler: Optional[nn.Module] = None  # callable returning a probability vector

# --------------------------------------------------------------------------- #
# Classical encoder / decoder
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

# --------------------------------------------------------------------------- #
# Hybrid network
# --------------------------------------------------------------------------- #
class HybridSamplerAutoencoderNet(nn.Module):
    """
    Encodes input data, optionally mixes it with a quantum sampler output,
    then decodes back to the original space.
    """
    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.sampler = cfg.sampler  # expects a callable returning probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.sampler is not None:
            # Quantum sampler is run in no‑grad mode to keep training classical
            with torch.no_grad():
                probs = self.sampler(z)
            probs = torch.tensor(probs, dtype=x.dtype, device=x.device)
            # Simple additive fusion; can be replaced by more sophisticated schemes
            z = z + probs
        return self.decoder(z)

# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def HybridSamplerAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    num_qubits: int = 4,
    sampler: Optional[nn.Module] = None,
) -> HybridSamplerAutoencoderNet:
    """
    Returns a fully‑configured HybridSamplerAutoencoderNet.
    The `sampler` should be a callable that accepts a latent tensor and
    returns a probability vector of size 2**num_qubits.
    """
    cfg = HybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_qubits=num_qubits,
        sampler=sampler,
    )
    return HybridSamplerAutoencoderNet(cfg)

__all__ = [
    "HybridConfig",
    "Encoder",
    "Decoder",
    "HybridSamplerAutoencoderNet",
    "HybridSamplerAutoencoder",
]
