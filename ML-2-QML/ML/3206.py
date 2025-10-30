"""Hybrid classical autoencoder with quantum latent regularization.

The module defines:

* :class:`EncoderNet` – a fully‑connected encoder.
* :class:`DecoderNet` – a fully‑connected decoder.
* :class:`AutoencoderHybridNet` – the full hybrid model.
* :func:`AutoencoderHybrid` – a factory mirroring the original seed.
* :func:`train_autoencoder_hybrid` – a simple training loop.

The quantum part is imported from :mod:`QuantumLatentLayer` (see the QML module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import the quantum latent layer defined in the QML module
try:
    from.QuantumLatentLayer import QuantumLatentLayer
except Exception:  # pragma: no cover
    # Fallback for standalone use – the class will be re‑defined in the QML file
    QuantumLatentLayer = None


# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderHybridConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # Quantum‑specific
    num_qubits: int = 4          # number of qubits used for the latent regularizer
    qnoise_level: float = 0.0    # optional quantum noise model (unused in this example)


# --------------------------------------------------------------------------- #
# 2. Classical encoder / decoder
# --------------------------------------------------------------------------- #
class EncoderNet(nn.Module):
    """Fully‑connected encoder."""
    def __init__(self, config: AutoencoderHybridConfig) -> None:
        super().__init__()
        layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderNet(nn.Module):
    """Fully‑connected decoder (mirrored architecture)."""
    def __init__(self, config: AutoencoderHybridConfig) -> None:
        super().__init__()
        layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# 3. Hybrid autoencoder
# --------------------------------------------------------------------------- #
class AutoencoderHybridNet(nn.Module):
    """Hybrid autoencoder that injects a quantum latent regularizer."""
    def __init__(self, config: AutoencoderHybridConfig) -> None:
        super().__init__()
        self.encoder = EncoderNet(config)
        # If QuantumLatentLayer is not available, raise an informative error.
        if QuantumLatentLayer is None:
            raise ImportError(
                "QuantumLatentLayer could not be imported. "
                "Ensure the QML module is available."
            )
        self.quantum = QuantumLatentLayer(
            latent_dim=config.latent_dim,
            num_qubits=config.num_qubits,
        )
        self.decoder = DecoderNet(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        q_latent = self.quantum(latent)
        return self.decoder(q_latent)


# --------------------------------------------------------------------------- #
# 4. Factory & training utilities
# --------------------------------------------------------------------------- #
def AutoencoderHybrid(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    num_qubits: int = 4,
) -> AutoencoderHybridNet:
    """Factory that returns a configured hybrid autoencoder."""
    config = AutoencoderHybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_qubits=num_qubits,
    )
    return AutoencoderHybridNet(config)


def train_autoencoder_hybrid(
    model: AutoencoderHybridNet,
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
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "AutoencoderHybrid",
    "AutoencoderHybridNet",
    "train_autoencoder_hybrid",
    "AutoencoderHybridConfig",
    "EncoderNet",
    "DecoderNet",
]
