"""
Hybrid classical autoencoder with optional quanvolutional feature extraction.

The network consists of:
  * A 2×2 patch convolution (QuanvolutionFilter) that mimics the 2×2 image patch kernel from the quanvolution example.
  * A fully‑connected encoder/decoder pair that maps to/from a latent space.
  * Configurable dropout and hidden layers.

This module is fully compatible with the original Autoencoder seed but adds richer feature extraction and an optional quantum kernel interface (for use with the QML module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# 1. Classical quanvolution filter (2×2 patch convolution)
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """
    Extracts 2×2 patches from single‑channel images and flattens them into a
    feature vector.  Mirrors the behaviour of the 2×2 convolution in the
    original quanvolution example.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C, H, W) with C == 1.
        Returns
        -------
        torch.Tensor
            Flattened patch features of shape (B, out_channels * (H//2) * (W//2)).
        """
        features = self.conv(x)
        return features.view(x.shape[0], -1)


# --------------------------------------------------------------------------- #
# 2. Autoencoder configuration
# --------------------------------------------------------------------------- #
@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_quanvolution: bool = True
    patch_channels: int = 1  # single‑channel images


# --------------------------------------------------------------------------- #
# 3. Hybrid autoencoder network
# --------------------------------------------------------------------------- #
class HybridAutoencoderNet(nn.Module):
    """
    Classical autoencoder that optionally uses a quanvolution filter for
    preprocessing before the dense encoder/decoder.
    """

    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Optional quanvolution feature extractor
        self.quanv = (
            QuanvolutionFilter(
                in_channels=config.patch_channels,
                out_channels=config.hidden_dims[0] // 2,
            )
            if config.use_quanvolution
            else None
        )

        # Encoder
        encoder_layers = []
        in_dim = (
            config.hidden_dims[0] // 2
            if config.use_quanvolution
            else config.input_dim
        )
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
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
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.quanv is not None:
            inputs = self.quanv(inputs)
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_quanvolution: bool = True,
    patch_channels: int = 1,
) -> HybridAutoencoderNet:
    """Factory that creates a fully‑configured hybrid autoencoder."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_quanvolution=use_quanvolution,
        patch_channels=patch_channels,
    )
    return HybridAutoencoderNet(config)


# --------------------------------------------------------------------------- #
# 4. Training helper
# --------------------------------------------------------------------------- #
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
    Simple reconstruction training loop.

    Parameters
    ----------
    model : HybridAutoencoderNet
        The autoencoder to train.
    data : torch.Tensor
        Input data of shape (N, D) or (N, C, H, W) if quanvolution is used.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini‑batch size.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularisation.
    device : torch.device | None
        Device to run on.  Defaults to CUDA if available.

    Returns
    -------
    list[float]
        Epoch‑wise training loss.
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


def _as_tensor(data: torch.Tensor | torch.Tensor | list | tuple) -> torch.Tensor:
    """Utility: ensure input is a float32 tensor on the CPU."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderNet",
    "HybridAutoencoderConfig",
    "train_hybrid_autoencoder",
]
