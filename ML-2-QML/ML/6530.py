"""Hybrid classical autoencoder with convolutional encoder and linear decoder.

This module extends the simple fully‑connected autoencoder to use a
convolutional feature extractor, inspired by the Conv filter from the
second reference pair.  It can be used as a drop‑in replacement for the
original Autoencoder class while providing a richer feature space.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class ConvFilter(nn.Module):
    """Simple 2‑D convolutional filter that mimics the quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and sigmoid activation."""
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

@dataclass
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    input_shape: Tuple[int, int]   # (H, W)
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    kernel_size: int = 2
    threshold: float = 0.0

class HybridAutoencoder(nn.Module):
    """Convolutional encoder + linear decoder autoencoder."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.input_shape = config.input_shape
        # Encoder: conv filter + flatten
        self.conv = ConvFilter(kernel_size=config.kernel_size, threshold=config.threshold)
        conv_out_h = config.input_shape[0] - config.kernel_size + 1
        conv_out_w = config.input_shape[1] - config.kernel_size + 1
        conv_out_dim = conv_out_h * conv_out_w
        # Linear encoder
        encoder_layers = []
        in_dim = conv_out_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Linear decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, conv_out_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input image into latent space."""
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to image shape."""
        out = self.decoder(z)
        out = out.view(-1, 1, *self.input_shape)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def HybridAutoencoderFactory(
    input_shape: Tuple[int, int],
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    kernel_size: int = 2,
    threshold: float = 0.0,
) -> HybridAutoencoder:
    """Convenience factory that mirrors the quantum helper."""
    cfg = HybridAutoencoderConfig(
        input_shape=input_shape,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        kernel_size=kernel_size,
        threshold=threshold,
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
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "HybridAutoencoderFactory",
    "train_hybrid_autoencoder",
]
