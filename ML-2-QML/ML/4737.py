"""Hybrid autoencoder – classical implementation.

This module extends the original fully‑connected autoencoder by embedding a
QCNN‑style feature extractor (``QCNNModel``) before the latent projection.
The decoder stays fully‑connected, allowing the model to be trained with
standard PyTorch pipelines.  The module also exposes a convenient
``kernel_matrix`` helper that derives from the classical RBF kernel
implementation in ``QuantumKernelMethod``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import the QCNN model defined in the seed QCNN.py
# The relative import works because the generated file lives in the same
# package as the original QCNN seed.
from.QCNN import QCNNModel


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
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1


class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder with QCNN‑style encoder and classical decoder."""

    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        # QCNN feature extractor works on 8‑dim inputs; for generality we
        # project the input to 8 dimensions first.
        self.preprocess = nn.Sequential(
            nn.Linear(config.input_dim, 8),
            nn.Tanh(),
        )
        self.qcnn = QCNNModel()

        # Map QCNN output to latent space
        self.encode = nn.Sequential(
            nn.Linear(8, config.latent_dim),
            nn.ReLU(),
        )

        # Classical decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in config.hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and decode a batch of inputs."""
        x = self.preprocess(x)
        x = self.qcnn(x)
        latent = self.encode(x)
        return self.decoder(latent)

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the latent representation."""
        x = self.preprocess(x)
        x = self.qcnn(x)
        return self.encode(x)


def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoder:
    """Convenience factory mirroring the original Autoencoder helper."""
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
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
    """Standard reconstruction training loop."""
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


# Kernel helpers – re‑exported from the classical RBF implementation
# to keep the API consistent with the quantum version.
from.QuantumKernelMethod import kernel_matrix as classical_kernel_matrix


__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
    "train_hybrid_autoencoder",
    "classical_kernel_matrix",
]
