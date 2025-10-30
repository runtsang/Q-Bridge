"""
QCNNAutoencoder: Classical hybrid model combining an autoencoder with a QCNN-inspired MLP.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple


# --------------------------------------------------------------------------- #
#  Autoencoder definition (adapted from the seed)
# --------------------------------------------------------------------------- #
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """A lightweight MLP autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
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
        """Encode inputs to latent space."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors back to input space."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 8,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory returning a configured autoencoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


# --------------------------------------------------------------------------- #
#  QCNN-inspired MLP (adapted from the seed)
# --------------------------------------------------------------------------- #
class QCNNModel(nn.Module):
    """Stack of fully‑connected layers emulating QCNN steps."""

    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning the configured QCNNModel."""
    return QCNNModel()


# --------------------------------------------------------------------------- #
#  Hybrid model: Autoencoder → QCNN
# --------------------------------------------------------------------------- #
class QCNNAutoencoder(nn.Module):
    """
    Hybrid model that first compresses the input via a classical autoencoder,
    then passes the latent representation through a QCNN-inspired MLP.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        ae_latent_dim: int = 8,
        ae_hidden_dims: Tuple[int, int] = (128, 64),
        ae_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            latent_dim=ae_latent_dim,
            hidden_dims=ae_hidden_dims,
            dropout=ae_dropout,
        )
        self.qcnn = QCNNModel(input_dim=ae_latent_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.autoencoder.encode(inputs)
        return self.qcnn(latent)


def QCNNAutoencoderFactory(
    input_dim: int,
    *,
    ae_latent_dim: int = 8,
    ae_hidden_dims: Tuple[int, int] = (128, 64),
    ae_dropout: float = 0.1,
) -> QCNNAutoencoder:
    """Factory returning a ready‑to‑train hybrid QCNN‑autoencoder."""
    return QCNNAutoencoder(
        input_dim=input_dim,
        ae_latent_dim=ae_latent_dim,
        ae_hidden_dims=ae_hidden_dims,
        ae_dropout=ae_dropout,
    )


__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "QCNNModel",
    "QCNN",
    "QCNNAutoencoder",
    "QCNNAutoencoderFactory",
]
