"""Hybrid autoencoder combining classical neural nets with a quantum decoder interface.

This module preserves the public API of the original Autoencoder while adding
a flag to trigger a quantum‑aware latent representation.  It is fully
importable and can be dropped into existing PyTorch pipelines.

Key components
---------------
- :class:`HybridAutoencoderConfig` – dataclass with architecture hyper‑parameters.
- :class:`HybridAutoencoderNet` – classic encoder/decoder with optional quantum flag.
- :func:`hybrid_autoencoder` – factory mirroring the original ``Autoencoder``.
- :func:`train_hybrid_autoencoder` – lightweight training loop that accepts an optional
  quantum decoder callable for end‑to‑end experiments.

The module intentionally keeps the classical training loop unchanged; the quantum
decoder is supplied externally (e.g. via :class:`qiskit_machine_learning.neural_networks.SamplerQNN`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, List, Optional

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
class HybridAutoencoderConfig:
    """Configuration values for :class:`HybridAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class HybridAutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder with optional quantum flag."""

    def __init__(self, config: HybridAutoencoderConfig, use_quantum: bool = False) -> None:
        super().__init__()
        self.use_quantum = use_quantum

        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (classical only)
        decoder_layers: List[nn.Module] = []
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
        """Return the latent representation."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector.  If ``use_quantum`` is True, the caller
        is responsible for transforming the output of the quantum decoder into
        the same shape expected by this method."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        return self.decode(latent)

def hybrid_autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_quantum: bool = False,
) -> HybridAutoencoderNet:
    """Factory that mirrors the original ``Autoencoder`` but supports a quantum flag."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoderNet(config, use_quantum=use_quantum)

def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    quantum_decoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> List[float]:
    """
    Simple reconstruction training loop returning the loss history.

    Parameters
    ----------
    model
        The :class:`HybridAutoencoderNet` instance.
    data
        Training data as a 2‑D tensor.
    quantum_decoder
        Optional callable that takes a latent tensor and returns a
        reconstruction.  If provided, the decoder’s output is used
        instead of the classical decoder during training.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            latent = model.encode(batch)
            # Use quantum decoder if supplied
            if quantum_decoder is not None:
                reconstruction = quantum_decoder(latent)
            else:
                reconstruction = model.decode(latent)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridAutoencoderConfig",
    "HybridAutoencoderNet",
    "hybrid_autoencoder",
    "train_hybrid_autoencoder",
]
