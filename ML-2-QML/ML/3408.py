"""Hybrid classical‑quantum autoencoder.

This module implements a lightweight fully‑connected autoencoder with an optional
quantum layer for the latent code.  The design is a direct synthesis of the
`Autoencoder.py` seed and the quantum‑full‑connected layer from
`QuantumNAT.py`.  The quantum processor is passed as a callable so that
the pure‑Python module remains free of quantum dependencies.

Usage
-----
>>> from hybrid_autoencoder import HybridAutoencoder, train_hybrid_autoencoder
>>> model = HybridAutoencoder(input_dim=784, latent_dim=32,
...                           hidden_dims=(256,128), dropout=0.2)
>>> history = train_hybrid_autoencoder(model, data, epochs=50)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Optional

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
class AutoencoderConfig:
    """Configuration values for :class:`HybridAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class HybridAutoencoderNet(nn.Module):
    """A fully‑connected autoencoder optionally wrapped with a quantum layer.

    Parameters
    ----------
    config: AutoencoderConfig
        Basic network configuration.
    quantum_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]]
        Callable that receives the latent tensor and returns a processed
        tensor of the same shape.  If ``None`` the identity function is used.
    """
    def __init__(
        self,
        config: AutoencoderConfig,
        quantum_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quantum_encoder = quantum_encoder or (lambda x: x)

        # Encoder
        encoder_layers: list[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: list[nn.Module] = []
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
        """Forward pass through the classical encoder."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classical decoder."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Complete autoencoder forward: classical encoder → quantum encoder →
        classical decoder."""
        latent = self.encode(inputs)
        quantum_latent = self.quantum_encoder(latent)
        return self.decode(quantum_latent)


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> HybridAutoencoderNet:
    """Factory that builds a hybrid autoencoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoderNet(config, quantum_encoder)


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
    """Training loop for the hybrid autoencoder.

    The loop is identical to the classical one but automatically routes data
    through the optional quantum layer.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
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
        if epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:03d}/{epochs:03d} – loss: {epoch_loss:.6f}")
    return history


__all__ = [
    "AutoencoderConfig",
    "HybridAutoencoderNet",
    "HybridAutoencoder",
    "train_hybrid_autoencoder",
]
