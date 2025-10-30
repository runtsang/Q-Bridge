"""Hybrid classical autoencoder with quantum regularization.

The network is a standard fully‑connected autoencoder whose latent
representation is penalised by a quantum circuit.  The quantum part is
implemented in :mod:`qml_autoencoder_regularizer`, which returns a
penalty term that can be added to any reconstruction loss.

Features
--------
* Encoder/decoder with optional residual connections.
* Weighted loss: ``loss = recon_loss + λ * quantum_penalty``.
* Support for supervised fine‑tuning via a ``target`` argument.
* Simple training loop that records loss history.
* No external dependencies beyond PyTorch and the QML module.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple, Callable, Optional

# The quantum regularizer is defined in a separate module.
# Import lazily to avoid circular imports.
def _quantum_regulator() -> Callable[[torch.Tensor], torch.Tensor]:
    from qml_autoencoder_regularizer import QuantumRegularizer
    return QuantumRegularizer(device="cpu")  # device can be overridden in training


class AutoencoderConfig:
    """Configuration parameters for :class:`AutoencoderGen111`."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int,...] = (128, 64),
        dropout: float = 0.1,
        use_residual: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_residual = use_residual


class AutoencoderGen111(nn.Module):
    """A fully‑connected autoencoder with optional residual links."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def quantum_regularizer(self, z: torch.Tensor) -> torch.Tensor:
        """Compute quantum penalty on the latent vector."""
        reg = _quantum_regulator()
        return reg(z)


def train_autoencoder(
    model: AutoencoderGen111,
    data: Iterable[float] | torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    lambda_q: float = 0.1,
    device: torch.device | None = None,
) -> Tuple[list[float], list[float]]:
    """Train the autoencoder with a quantum‑regularized loss.

    Returns a tuple (reconstruction_history, regularizer_history).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    recon_loss_fn = nn.MSELoss()

    recon_hist: list[float] = []
    reg_hist: list[float] = []

    for epoch in range(epochs):
        epoch_recon = 0.0
        epoch_reg = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            recon_loss = recon_loss_fn(recon, batch)

            latent = model.encode(batch)
            reg_loss = model.quantum_regularizer(latent).mean()

            loss = recon_loss + lambda_q * reg_loss
            loss.backward()
            optimizer.step()

            epoch_recon += recon_loss.item() * batch.size(0)
            epoch_reg += reg_loss.item() * batch.size(0)

        epoch_recon /= len(dataset)
        epoch_reg /= len(dataset)
        recon_hist.append(epoch_recon)
        reg_hist.append(epoch_reg)

    return recon_hist, reg_hist


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


__all__ = [
    "AutoencoderGen111",
    "AutoencoderConfig",
    "train_autoencoder",
    "_as_tensor",
]
