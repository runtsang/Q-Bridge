"""
Hybrid convolutional autoencoder – classical implementation.

The module defines two public classes:
- ``ConvGenAutoencoder``  – a PyTorch model that first applies a learnable
  convolutional filter (``ConvFilter``) and then passes the flattened
  activations through a lightweight MLP autoencoder.
- ``AutoencoderNet`` – retained from the original Autoencoder seed but
  refactored to accept any hidden configuration.

The design keeps the original Conv API (``Conv()`` factory) while
expanding it to work with multi‑channel inputs and a configurable
kernel size.  The autoencoder is fully compatible with the original
training loop but now accepts 2‑D image tensors instead of flat vectors.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple

__all__ = ["Conv", "ConvFilter", "ConvGenAutoencoder", "AutoencoderNet", "AutoencoderConfig", "train_autoencoder"]


# --------------------------------------------------------------------------- #
# 1. Classical convolutional filter – drop‑in replacement for the quantum filter
# --------------------------------------------------------------------------- #
def Conv(kernel_size: int = 3, threshold: float = 0.0) -> nn.Module:
    """Return a callable convolutional filter that emulates the quantum quanvolution.

    The filter is a single‑channel 2‑D convolution followed by a sigmoid
    activation that is threshold‑adjusted.  It is intentionally lightweight
    so that it can be inserted as the first layer of a larger network.
    """
    class ConvFilter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            # Single‑channel filter with bias – equivalent to the quantum circuit
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : torch.Tensor
                Input image tensor of shape (B, C=1, H, W).

            Returns
            -------
            torch.Tensor
                Filtered activations of shape (B, 1, H-k+1, W-k+1).
            """
            logits = self.conv(x)
            activations = torch.sigmoid(logits - self.threshold)
            return activations

    return ConvFilter()


# --------------------------------------------------------------------------- #
# 2. Autoencoder backbone – lightweight MLP
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Multilayer perceptron autoencoder.

    The architecture mirrors the original seed but is now wrapped in a
    dataclass for easy configuration.  It can be used stand‑alone or as
    part of the hybrid model below.
    """
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._make_mlp(
            input_dim=cfg.input_dim,
            hidden_dims=cfg.hidden_dims,
            output_dim=cfg.latent_dim,
            dropout=cfg.dropout,
        )
        self.decoder = self._make_mlp(
            input_dim=cfg.latent_dim,
            hidden_dims=tuple(reversed(cfg.hidden_dims)),
            output_dim=cfg.input_dim,
            dropout=cfg.dropout,
        )

    @staticmethod
    def _make_mlp(input_dim: int, hidden_dims: Tuple[int,...], output_dim: int,
                  dropout: float) -> nn.Sequential:
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def Autoencoder(cfg: AutoencoderConfig) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# 3. Hybrid convolutional autoencoder
# --------------------------------------------------------------------------- #
class ConvGenAutoencoder(nn.Module):
    """Convolutional autoencoder that uses a learnable filter before the dense backbone.

    The first layer is the ``Conv`` filter from this module; the subsequent
    layers are the MLP autoencoder.  The module is fully differentiable
    and can be trained with the same ``train_autoencoder`` loop.
    """
    def __init__(self, kernel_size: int = 3, threshold: float = 0.0,
                 hidden_dims: Tuple[int,...] = (128, 64),
                 latent_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.filter = Conv(kernel_size=kernel_size, threshold=threshold)
        # The filter reduces spatial dimensions; compute flattened size
        dummy = torch.zeros(1, 1, 28, 28)  # assume MNIST‑style input
        filtered = self.filter(dummy)
        flat_dim = filtered.numel()
        cfg = AutoencoderConfig(
            input_dim=flat_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.autoencoder = Autoencoder(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.filter(x)
        x = x.view(x.size(0), -1)
        return self.autoencoder(x)


def train_autoencoder(
    model: nn.Module,
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
    model = model.to(device)
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


def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor
