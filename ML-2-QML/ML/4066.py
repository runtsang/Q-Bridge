"""
HybridAutoencoder – Classical fully‑connected autoencoder with training utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterable, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


@dataclass
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class HybridAutoencoder(nn.Module):
    """
    Fully‑connected autoencoder with configurable depth, dropout and a
    convenient training routine.

    The interface matches the quantum counterpart: ``encode``, ``decode`` and
    ``forward`` are defined so that the class can be dropped into a hybrid
    pipeline.
    """

    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        encoder = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder.append(nn.Linear(in_dim, h))
            encoder.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder.append(nn.Dropout(config.dropout))
            in_dim = h
        encoder.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder.append(nn.Linear(in_dim, h))
            decoder.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder.append(nn.Dropout(config.dropout))
            in_dim = h
        decoder.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def train_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """
    Train the autoencoder on the provided tensor data.

    Parameters
    ----------
    model : HybridAutoencoder
        The model to optimise.
    data : torch.Tensor
        Training data of shape ``(N, input_dim)``.
    epochs : int
        Number of training epochs.
    batch_size : int
        Size of each minibatch.
    lr : float
        Learning rate for Adam.
    weight_decay : float
        Optional L2 regularisation.
    device : torch.device | None
        Target device; defaults to CUDA if available.

    Returns
    -------
    history : List[float]
        List of epoch‑average reconstruction losses (MSE).
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history


__all__ = ["HybridAutoencoder", "HybridAutoencoderConfig", "train_autoencoder"]
