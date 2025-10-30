"""
Hybrid classical neural network combining QCNN, Sampler, Autoencoder, and Quanvolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


# ---------- Sub‑components ----------
class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating the quantum convolution steps."""

    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
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


class SamplerQNN(nn.Module):
    """Softmax sampler mimicking a quantum sampler output."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))


class QuanvolutionFilter(nn.Module):
    """Classical convolutional filter inspired by the quanvolution example."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the quanvolution filter followed by a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


# ---------- Hybrid network ----------
class HybridQCNN(nn.Module):
    """
    End‑to‑end network that:
      1. Extracts 2×2 patches with a quanvolution filter.
      2. Projects to 8‑dimensional space for a QCNN encoder.
      3. Applies an autoencoder bottleneck.
      4. Maps to a 2‑dimensional sampler output.
    """

    def __init__(
        self,
        input_dim: int = 784,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.quanvolution = QuanvolutionFilter()
        self.linear_to_qcnn = nn.Linear(4 * 14 * 14, 8)
        self.qcnn = QCNNModel(input_dim=8)
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=8,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )
        self.latent_to_sampler = nn.Linear(latent_dim, 2)
        self.sampler = SamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quanvolution(x)
        x = self.linear_to_qcnn(x)
        x = self.qcnn(x)
        x = self.autoencoder.encode(x)
        x = self.latent_to_sampler(x)
        return self.sampler(x)


__all__ = ["HybridQCNN"]
