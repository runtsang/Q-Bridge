"""
UnifiedSamplerAutoencoder: hybrid classical‑quantum architecture.
Implements a two‑stage pipeline:
  1. Classical sampler network (2→4→2) for probability vector generation.
  2. Autoencoder that compresses the sampler output into a latent vector,
     optionally feeding it to a quantum sampler for post‑processing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

# --------------------------------------------------------------------------- #
#  Classical sampler: 2‑dim input → 4‑dim hidden → 2‑dim output
# --------------------------------------------------------------------------- #
class ClassicalSampler(nn.Module):
    """
    A lightweight, fully‑connected network that mirrors the original
    SamplerQNN class.  It is intentionally kept small so that the
    network can be fused with the quantum part without adding excessive
    compute.
    """
    def __init__(self, hidden_dim: int = 4, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softmax over the last dimension to obtain a probability vector
        return F.softmax(self.net(x), dim=-1)

# --------------------------------------------------------------------------- #
#  Autoencoder: 2‑dim input (the sampler output) → latent → reconstructions
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for the autoencoder part."""
    input_dim: int = 2
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """A standard fully‑connected autoencoder based on the seed."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        encoder_layers: List[torch.nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[torch.nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

# --------------------------------------------------------------------------- #
#  Unified sampler autoencoder
# --------------------------------------------------------------------------- #
class UnifiedSamplerAutoencoder(nn.Module):
    """
    Combines a classical sampler with an autoencoder.
    Forward pass: input -> sampler -> latent -> reconstruction.
    """
    def __init__(self,
                 sampler_hidden: int = 4,
                 ae_cfg: AutoencoderConfig | None = None) -> None:
        super().__init__()
        self.sampler = ClassicalSampler(hidden_dim=sampler_hidden)
        self.ae_cfg = ae_cfg or AutoencoderConfig()
        self.autoencoder = AutoencoderNet(self.ae_cfg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            latent: tensor of shape (batch, latent_dim)
            reconstruction: tensor of shape (batch, input_dim)
        """
        probs = self.sampler(x)
        latent = self.autoencoder.encode(probs)
        reconstruction = self.autoencoder.decode(latent)
        return latent, reconstruction

# --------------------------------------------------------------------------- #
#  Training helper
# --------------------------------------------------------------------------- #
def train_autoencoder(model: AutoencoderNet,
                      data: torch.Tensor,
                      epochs: int = 100,
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      weight_decay: float = 0.0,
                      device: torch.device | None = None) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
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

# --------------------------------------------------------------------------- #
#  Helper to convert to tensor
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = ["UnifiedSamplerAutoencoder", "train_autoencoder", "AutoencoderConfig", "AutoencoderNet", "ClassicalSampler"]
