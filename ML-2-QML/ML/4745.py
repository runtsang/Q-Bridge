from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder used as a feature extractor."""
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        encoder = []
        d = input_dim
        for h in hidden_dims:
            encoder.append(nn.Linear(d, h))
            encoder.append(nn.ReLU())
            if dropout > 0.0:
                encoder.append(nn.Dropout(dropout))
            d = h
        encoder.append(nn.Linear(d, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        d = latent_dim
        for h in reversed(hidden_dims):
            decoder.append(nn.Linear(d, h))
            decoder.append(nn.ReLU())
            if dropout > 0.0:
                decoder.append(nn.Dropout(dropout))
            d = h
        decoder.append(nn.Linear(d, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class QCNNModel(nn.Module):
    """Classical QCNN architecture mirroring quantum convolution‑pooling."""
    def __init__(self, feature_dim: int = 32, num_classes: int = 1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(feature_dim, 64), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(64, 48), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(48, 32), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(32, 16), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.head = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridQCNN(nn.Module):
    """Hybrid model: autoencoder → QCNN."""
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1, num_classes: int = 1):
        super().__init__()
        self.autoencoder = AutoencoderNet(input_dim, latent_dim, hidden_dims, dropout)
        self.qcnn = QCNNModel(feature_dim=latent_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.autoencoder.encode(x)
        return self.qcnn(latent)

def HybridQCNNFactory(
    input_dim: int,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    num_classes: int = 1,
) -> HybridQCNN:
    """Return a fully configured hybrid QCNN."""
    return HybridQCNN(input_dim, latent_dim, hidden_dims, dropout, num_classes)

__all__ = ["AutoencoderNet", "QCNNModel", "HybridQCNN", "HybridQCNNFactory"]
