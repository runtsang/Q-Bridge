"""Hybrid QCNN–Autoencoder model, classical implementation."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


class AutoencoderConfig:
    """Configuration values for the latent MLP."""
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dims: Tuple[int,...] = (64, 32), dropout: float = 0.05):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
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

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


class QCNNHybrid(nn.Module):
    """Classical hybrid QCNN with an autoencoder bottleneck."""
    def __init__(self, input_dim: int = 8, latent_dim: int = 16) -> None:
        super().__init__()

        # Feature extraction – mimics the QCNN convolution & pooling
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh()
        )

        # Autoencoder bottleneck
        ae_cfg = AutoencoderConfig(input_dim=4, latent_dim=latent_dim)
        self.autoencoder = AutoencoderNet(ae_cfg)

        # Classifier head on the latent representation
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple:
          (logits, reconstruction)
        """
        features = self.feature_map(inputs)
        latent = self.autoencoder.encode(features)
        logits = torch.sigmoid(self.classifier(latent))
        reconstruction = self.autoencoder.decode(latent)
        return logits, reconstruction


def QCNNHybridModel(input_dim: int = 8, latent_dim: int = 16) -> QCNNHybrid:
    """Factory returning the configured hybrid QCNN model."""
    return QCNNHybrid(input_dim=input_dim, latent_dim=latent_dim)


__all__ = ["QCNNHybrid", "QCNNHybridModel", "AutoencoderConfig", "AutoencoderNet"]
