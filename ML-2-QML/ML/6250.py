"""Hybrid classical autoencoder with optional quantum refinement layer.

This module implements a classical autoencoder that can optionally plug in a
quantum refinement layer.  The classical core mirrors the original AutoencoderNet
architecture from the seed, but exposes a `quantum_layer` callable that defaults
to the identity.  Users can replace this callable with a quantum implementation
(e.g. the one provided in the QML module) to obtain a hybrid model.

The training loop is identical to the original, but accepts the hybrid model
and automatically forwards latent vectors through the quantum layer if
supplied.  The design keeps the ML side entirely classical (NumPy, PyTorch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class HybridAutoEncoderConfig:
    """Configuration values for :class:`HybridAutoEncoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    quantum_layer: Callable[[torch.Tensor], torch.Tensor] | None = None


class HybridAutoEncoder(nn.Module):
    """A hybrid autoencoder that optionally applies a quantum refinement layer.

    The classical core mirrors the original AutoencoderNet, while the quantum
    layer is a pluggable callable that can be replaced by a QML implementation.
    """
    def __init__(self, config: HybridAutoEncoderConfig) -> None:
        super().__init__()
        self.encoder = self._build_encoder(config)
        self.decoder = self._build_decoder(config)
        self.quantum_layer = config.quantum_layer or (lambda x: x)  # identity

    def _build_encoder(self, config: HybridAutoEncoderConfig) -> nn.Sequential:
        layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.latent_dim))
        return nn.Sequential(*layers)

    def _build_decoder(self, config: HybridAutoEncoderConfig) -> nn.Sequential:
        layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.input_dim))
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def refine(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply the quantum refinement layer."""
        return self.quantum_layer(latents)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z = self.encode(inputs)
        z_refined = self.refine(z)
        return self.decode(z_refined)


def train_hybrid_autoencoder(
    model: HybridAutoEncoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop that accepts a hybrid model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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


__all__ = ["HybridAutoEncoder", "HybridAutoEncoderConfig", "train_hybrid_autoencoder"]
