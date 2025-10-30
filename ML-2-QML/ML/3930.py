"""Hybrid classical auto‑encoder combining QCNN feature extraction and a fully‑connected bottleneck.

The network mirrors the QCNN model for the feature map, followed by an
auto‑encoder that compresses to a latent space and reconstructs the input.
The design allows easy integration into existing PyTorch workflows and
provides a concrete target for quantum‑classical hybrid training.

Author: gpt-oss-20b
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data) -> torch.Tensor:
    """Coerce any numeric array to a float32 tensor on the current device."""
    return torch.as_tensor(data, dtype=torch.float32, device=torch.device("cpu"))


@dataclass
class HybridAutoencoderConfig:
    """Hyper‑parameters for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    conv_features: Tuple[int,...] = (16, 16, 12, 8, 4, 4)  # QCNN‑style feature shape


class HybridAutoencoder(nn.Module):
    """A fully‑connected auto‑encoder preceded by a QCNN‑style feature extractor."""

    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        # Feature extractor – mimics QCNN with linear + Tanh layers
        self.feature_map = nn.Sequential(
            *[nn.Sequential(nn.Linear(cfg.input_dim if i == 0 else cfg.conv_features[i - 1],
                                      f, nn.Tanh())) for i, f in enumerate(cfg.conv_features)]
        )

        # Encoder
        encoder_layers = []
        in_dim = cfg.conv_features[-1]
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.conv_features[-1]))
        self.decoder = nn.Sequential(*decoder_layers)

        # Re‑mapping to original dimensionality
        self.reconstruction = nn.Linear(cfg.conv_features[-1], cfg.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode → decode → reconstruct."""
        x = self.feature_map(x)
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return self.reconstruction(decoded)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation."""
        x = self.feature_map(x)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent vector."""
        decoded = self.decoder(z)
        return self.reconstruction(decoded)


def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoder:
    """Convenience factory mirroring the original Autoencoder helper."""
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
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
    """Standard reconstruction training loop."""
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
    "train_hybrid_autoencoder",
]
