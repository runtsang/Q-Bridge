"""Hybrid classical auto‑encoder with optional kernel regularisation.

This module merges the fully‑connected auto‑encoder from the seed with
the classical radial‑basis‑function kernel utilities.  The `HybridAutoencoder`
class exposes an API identical to the original `AutoencoderNet` but adds a
`kernel_regularizer` flag that, when set, adds an RBF‑kernel based loss
term to the reconstruction loss.  The implementation stays entirely on
PyTorch / NumPy and can be used on CPU or GPU.

The class name `HybridAutoencoder` is chosen to indicate the fusion of
classical auto‑encoding with quantum‑kernel inspired regularisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["HybridAutoencoder", "AutoencoderConfig", "train_hybrid_autoencoder"]

# --------------------------------------------------------------------------- #
# Configuration & utilities
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.0
    kernel_regularizer: bool = False
    gamma: float = 1.0

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

# --------------------------------------------------------------------------- #
# Classical RBF kernel
# --------------------------------------------------------------------------- #

def rbf_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float) -> np.ndarray:
    """Return the Gram matrix using a classical RBF kernel."""
    a_np = np.array([x.cpu().numpy() for x in a])
    b_np = np.array([y.cpu().numpy() for y in b])
    diff = a_np[:, None, :] - b_np[None, :, :]
    return np.exp(-gamma * np.sum(diff ** 2, axis=-1))

# --------------------------------------------------------------------------- #
# Hybrid auto‑encoder
# --------------------------------------------------------------------------- #

class HybridAutoencoder(nn.Module):
    """A fully‑connected auto‑encoder that optionally regularises with a kernel."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

    # ----------------------------------------------------------------------- #
    # Kernel‑based regularisation
    # ----------------------------------------------------------------------- #

    def _kernel_loss(self, latents: torch.Tensor) -> torch.Tensor:
        """Classical RBF kernel similarity loss on latent space."""
        x = latents.detach().cpu().numpy()
        K = rbf_kernel_matrix(x, x, self.config.gamma)
        mask = ~np.eye(K.shape[0], dtype=bool)
        loss = K[mask].mean()
        return torch.tensor(loss, device=latents.device, dtype=latents.dtype)

# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #

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
    """Train the hybrid auto‑encoder, optionally with kernel regularisation."""
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
            if model.config.kernel_regularizer:
                latent = model.encode(batch)
                loss += model._kernel_loss(latent)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history
