"""Hybrid classical neural network combining regression, sampling, autoencoding, and convolutional patterns."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------
# 1. Estimator network (regression)
# --------------------------------------------------------------------
class EstimatorNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------
# 2. Sampler network (categorical distribution)
# --------------------------------------------------------------------
class SamplerModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

# --------------------------------------------------------------------
# 3. Autoencoder (compress‑then‑reconstruct)
# --------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# --------------------------------------------------------------------
# 4. QCNN (convolution‑inspired feature extractor)
# --------------------------------------------------------------------
class QCNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# --------------------------------------------------------------------
# 5. Hybrid model
# --------------------------------------------------------------------
class HybridNet(nn.Module):
    """
    Composite model that first compresses the input via an autoencoder,
    then extracts hierarchical features with QCNN, and finally produces
    a regression output (Estimator) and a categorical distribution (Sampler).
    """
    def __init__(self, input_dim: int = 2) -> None:
        super().__init__()
        self.encoder = AutoencoderNet(AutoencoderConfig(input_dim=input_dim))
        self.qcnn = QCNNModel()
        self.estimator = EstimatorNN()
        self.sampler = SamplerModule()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder.encode(x)
        feat = self.qcnn(z)
        out = self.estimator(feat)
        prob = self.sampler(feat)
        return out, prob

def HybridEstimator(input_dim: int = 2) -> HybridNet:
    """Factory returning a fully‑configured hybrid model."""
    return HybridNet(input_dim)

__all__ = [
    "EstimatorNN",
    "SamplerModule",
    "AutoencoderConfig",
    "AutoencoderNet",
    "QCNNModel",
    "HybridNet",
    "HybridEstimator",
]
