"""
Hybrid autoencoder sampler for classical training.

This module combines the fully‑connected autoencoder architecture from
reference pair 2 with the probabilistic sampler interface of the
SamplerQNN helper (reference pair 1).  The network can optionally
output a softmax over the latent space, providing a classical
probabilistic sampler that mirrors the quantum SamplerQNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HybridAutoSamplerConfig:
    """
    Configuration for :class:`HybridAutoSampler`.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vectors.
    latent_dim : int, default 32
        Size of the latent bottleneck.
    hidden_dims : Tuple[int, int], default (128, 64)
        Sizes of the hidden layers in encoder/decoder.
    dropout : float, default 0.1
        Dropout probability applied after each ReLU.
    sampler : bool, default False
        If True the forward pass returns a softmax over the latent
        representation instead of a decoded reconstruction.
    """

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    sampler: bool = False


class HybridAutoSampler(nn.Module):
    """
    A lightweight autoencoder that can act as a classical sampler.

    The network is symmetric: an encoder maps the input to the latent
    space, and a decoder reconstructs the input from the latent
    representation.  When ``config.sampler`` is True, the forward
    method returns a probability distribution over the latent
    dimension instead of a reconstruction, providing a classical
    sampling interface that parallels the quantum SamplerQNN.
    """

    def __init__(self, config: HybridAutoSamplerConfig) -> None:
        super().__init__()
        self.config = config

        # Build encoder
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

        # Build decoder
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map inputs to the latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct inputs from the latent space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass.

        If ``config.sampler`` is True, return a softmax over the latent
        representation to emulate probabilistic sampling.  Otherwise,
        return the decoded reconstruction.
        """
        z = self.encode(x)
        if self.config.sampler:
            return F.softmax(z, dim=-1)
        return self.decode(z)


def HybridAutoSamplerFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    sampler: bool = False,
) -> HybridAutoSampler:
    """Convenience factory mirroring the seed style."""
    cfg = HybridAutoSamplerConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        sampler=sampler,
    )
    return HybridAutoSampler(cfg)


def train_hybrid_autoencoder(
    model: HybridAutoSampler,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Simple reconstruction training loop for the hybrid autoencoder.

    Parameters
    ----------
    model : HybridAutoSampler
        The network to train.
    data : torch.Tensor
        Dataset of shape (N, input_dim).
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini‑batch size.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularisation strength.
    device : torch.device | None
        Target device; defaults to CUDA if available.

    Returns
    -------
    list[float]
        History of training loss per epoch.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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


def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = ["HybridAutoSampler", "HybridAutoSamplerFactory", "train_hybrid_autoencoder"]
