"""
AutoencoderGen343 – Classical component

This module defines a flexible MLP autoencoder that can optionally
delegate the latent representation to a user‑supplied quantum
module.  The interface is deliberately minimal to keep it
interoperable with the quantum helper located in
``Autoencoder__gen343_qml.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Coerce input to a float32 tensor on the default device."""
    tensor = (
        data if isinstance(data, torch.Tensor) else torch.as_tensor(data, dtype=torch.float32)
    )
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderConfig:
    """Hyper‑parameters for :class:`HybridAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1


class HybridAutoencoderNet(nn.Module):
    """
    An MLP autoencoder that can optionally forward its latent vector
    through a quantum circuit supplied at runtime.

    Parameters
    ----------
    config : AutoencoderConfig
        Hyper‑parameters.
    quantum_encoder : Callable[[torch.Tensor], torch.Tensor] | None
        Function that accepts a latent tensor and returns a refined latent.
        If ``None`` the classical encoder is used as is.
    """

    def __init__(
        self,
        config: AutoencoderConfig,
        quantum_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quantum_encoder = quantum_encoder

        # Classical encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Classical decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        latents = self.encoder(inputs)
        if self.quantum_encoder is not None:
            # The quantum encoder is expected to be a black‑box that returns a
            # refined latent of the same shape.
            latents = self.quantum_encoder(latents)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    quantum_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> HybridAutoencoderNet:
    """Convenience factory mirroring the quantum helper."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoderNet(config, quantum_encoder=quantum_encoder)


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
    Train loop that supports both classical and hybrid quantum encoders.
    The loss is the MSE between reconstructions and inputs.
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
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "AutoencoderConfig",
    "HybridAutoencoderNet",
    "HybridAutoencoder",
    "train_hybrid_autoencoder",
]
