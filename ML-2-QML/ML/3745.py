"""Hybrid classical autoencoder with a quanvolutional encoder.

The class combines a conventional 2×2 convolutional filter (as in the
original `QuanvolutionFilter`) with a fully‑connected decoder.  It can be
trained on MNIST‑like data to reconstruct inputs, providing a baseline for
quantum‑inspired experiments.

The API mirrors the classical `AutoencoderNet` from the reference seed, but
the encoder is a 2×2 convolution that flattens to a 4×14×14 feature map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable

# --------------------------------------------------------------------------- #
# Classical encoder – simple 2x2 convolution
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """
    Classical 2×2 filter that slides over the input image and produces
    a reduced feature map.  Each 2×2 patch is mapped to a 4‑dimensional
    feature vector via a 1×1 convolution.
    """
    def __init__(self) -> None:
        super().__init__()
        # 1 input channel → 4 output channels, kernel 2, stride 2
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Feature map of shape (B, 4*14*14).
        """
        features = self.conv(x)  # (B, 4, 14, 14)
        return features.view(x.size(0), -1)  # (B, 784)


# --------------------------------------------------------------------------- #
# Configuration dataclass
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """
    Configuration for the fully‑connected decoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the encoded feature vector (default 784).
    latent_dim : int
        Size of the latent representation (default 32).
    hidden_dims : Tuple[int, int]
        Hidden layer sizes for the decoder (default (128, 64)).
    dropout : float
        Dropout probability (default 0.1).
    """
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


# --------------------------------------------------------------------------- #
# Fully‑connected decoder
# --------------------------------------------------------------------------- #
class DecoderNet(nn.Module):
    """
    Decoder that maps encoded features back to the original image space.
    """
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.latent_dim))  # latent vector
        self.encoder = nn.Sequential(*layers)

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
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.encode(x)
        return self.decode(z)


# --------------------------------------------------------------------------- #
# Main autoencoder module
# --------------------------------------------------------------------------- #
class QuanvolutionAutoencoder(nn.Module):
    """
    Hybrid autoencoder that first applies a classical quanvolution filter
    and then feeds the flattened features into a fully‑connected decoder.
    """
    def __init__(self,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        config = AutoencoderConfig(
            input_dim=4 * 14 * 14,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.decoder = DecoderNet(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape (B, 1, 28, 28).
        """
        features = self.qfilter(x)  # (B, 784)
        recon_flat = self.decoder(features)  # (B, 784)
        recon = recon_flat.view(x.size(0), 1, 28, 28)
        return recon


# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def train_autoencoder(
    model: QuanvolutionAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Simple reconstruction training loop returning the loss history.

    Parameters
    ----------
    model : QuanvolutionAutoencoder
        The autoencoder to train.
    data : torch.Tensor
        Dataset of shape (N, 1, 28, 28).
    epochs : int, optional
        Number of training epochs, by default 100.
    batch_size : int, optional
        Mini‑batch size, by default 64.
    lr : float, optional
        Learning rate, by default 1e-3.
    weight_decay : float, optional
        L2 weight decay, by default 0.0.
    device : torch.device | None, optional
        Training device; defaults to CUDA if available.

    Returns
    -------
    list[float]
        List of epoch‑level reconstruction losses.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = []

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


__all__ = ["QuanvolutionAutoencoder", "train_autoencoder"]
