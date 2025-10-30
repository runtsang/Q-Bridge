"""
QuanvolutionAutoencoder: Classical hybrid of a quanvolution filter and a
fully‑connected autoencoder.  The filter operates on 2×2 patches of a 28×28
image, producing a feature map that is then fed into an MLP encoder/decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


class QuanvolutionFilter(nn.Module):
    """
    Extracts 2×2 patches from a 28×28 single‑channel image, applies a 2‑qubit
    quantum kernel (here simulated with a random linear layer for speed) and
    flattens the result into a 1‑D feature vector.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4) -> None:
        super().__init__()
        # Classical surrogate for the quantum kernel
        self.kernel = nn.Linear(in_channels * 4, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of images with shape (B, C, H, W).  Expected to be 28×28.
        Returns
        -------
        torch.Tensor
            Feature vector of shape (B, 4 * 14 * 14).
        """
        B, C, H, W = x.shape
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, C, 14, 14, 2, 2)
        patches = patches.contiguous().view(B, C, 14, 14, 4)
        patches = patches.permute(0, 2, 3, 1, 4).reshape(B, 14 * 14, C * 4)
        features = self.kernel(patches)  # (B, 14*14, out_channels)
        return features.view(B, -1)


@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class QuanvolutionAutoencoderNet(nn.Module):
    """
    Encoder: quanvolution filter → linear encoder.
    Decoder: linear decoder → reconstruction.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


def QuanvolutionAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> QuanvolutionAutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return QuanvolutionAutoencoderNet(config)


def train_quanvolution_autoencoder(
    model: QuanvolutionAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Simple MSE reconstruction training loop.
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = [
    "QuanvolutionFilter",
    "AutoencoderConfig",
    "QuanvolutionAutoencoderNet",
    "QuanvolutionAutoencoder",
    "train_quanvolution_autoencoder",
]
