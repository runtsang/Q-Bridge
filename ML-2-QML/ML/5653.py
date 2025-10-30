"""
Hybrid classical‑quantum autoencoder.

The module defines a PyTorch neural network that optionally routes the latent
representation through a quantum fully‑connected layer (FCL).  The quantum
layer is supplied by the `FCL` function from the classical stand‑in or the
real Qiskit implementation.  The network remains fully differentiable on the
classical side; the quantum block is treated as a black‑box that returns a
numpy array, which is converted back to a torch tensor.

Typical usage::

    from HybridAutoencoder import HybridAutoencoder, train_hybrid_autoencoder
    model = HybridAutoencoder(input_dim=784, latent_dim=32, quantum_layer=FCL())
    history = train_hybrid_autoencoder(model, data, epochs=50)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    quantum_layer: Optional[Callable[[np.ndarray], np.ndarray]] = None

# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert data to a float32 torch tensor on the current device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

# --------------------------------------------------------------------------- #
# Core network
# --------------------------------------------------------------------------- #
class HybridAutoencoderNet(nn.Module):
    """A classical autoencoder with an optional quantum latent processor."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_mlp(
            in_dim=config.input_dim,
            out_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
        self.decoder = self._build_mlp(
            in_dim=config.latent_dim,
            out_dim=config.input_dim,
            hidden_dims=config.hidden_dims[::-1],
            dropout=config.dropout,
        )
        self.quantum_layer = config.quantum_layer

    @staticmethod
    def _build_mlp(in_dim: int, out_dim: int, hidden_dims: Tuple[int,...], dropout: float) -> nn.Sequential:
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        if self.quantum_layer is not None:
            # Convert to numpy for the quantum block
            z_np = z.detach().cpu().numpy()
            # Run quantum layer on each sample in the batch
            z_q = np.array([self.quantum_layer(sample) for sample in z_np])
            z = torch.from_numpy(z_q).to(x.device).float()
        return self.decode(z)

# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_layer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> HybridAutoencoderNet:
    """Return a configured :class:`HybridAutoencoderNet`."""
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum_layer=quantum_layer,
    )
    return HybridAutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Train the hybrid autoencoder.

    Parameters
    ----------
    model : HybridAutoencoderNet
        The network to train.
    data : torch.Tensor
        Training data of shape (N, input_dim).
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini‑batch size.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay for Adam.
    device : torch.device | None
        Training device, defaults to CUDA if available.

    Returns
    -------
    history : list[float]
        Reconstruction loss per epoch.
    """
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

__all__ = ["HybridAutoencoder", "HybridAutoencoderNet", "train_hybrid_autoencoder", "HybridAutoencoderConfig"]
