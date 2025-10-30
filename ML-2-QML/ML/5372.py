"""Importable ML module defining HybridBinaryClassifier.

This module extends the original ClassicalQuantumBinaryClassification
by adding optional autoencoder compression and a lightweight
EstimatorQNN head.  All components are purely classical and
compatible with PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# Utility: tensor conversion
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

# --------------------------------------------------------------------------- #
# Auto‑encoder (from reference 3)
# --------------------------------------------------------------------------- #
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory returning a configured auto‑encoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)

# --------------------------------------------------------------------------- #
# Simple convolutional filter (from reference 2)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Emulates a quantum filter with a 2‑D convolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

def Conv() -> ConvFilter:
    """Return a callable instance of the filter."""
    return ConvFilter()

# --------------------------------------------------------------------------- #
# Estimator‑style regressor (from reference 4)
# --------------------------------------------------------------------------- #
class EstimatorQNNNet(nn.Module):
    """Fully‑connected regression network mirroring the EstimatorQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

def EstimatorQNN() -> EstimatorQNNNet:
    """Factory returning the estimator network."""
    return EstimatorQNNNet()

# --------------------------------------------------------------------------- #
# Main hybrid classifier
# --------------------------------------------------------------------------- #
class HybridBinaryClassifier(nn.Module):
    """
    Convolutional feature extractor + optional auto‑encoder + classical or
    estimator head.  The architecture mirrors the original QCNet but
    provides knobs to switch on/off the quantum‑style layers.
    """
    def __init__(
        self,
        use_autoencoder: bool = False,
        use_estimator: bool = False,
        autoencoder_config: AutoencoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.use_autoencoder = use_autoencoder
        self.use_estimator = use_estimator

        # Convolutional backbone (same as the original QCNet)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Optional auto‑encoder
        if self.use_autoencoder:
            cfg = (
                autoencoder_config
                or AutoencoderConfig(input_dim=84, latent_dim=32, hidden_dims=(64, 32), dropout=0.1)
            )
            self.autoencoder = Autoencoder(cfg.input_dim)
        else:
            self.autoencoder = None

        # Classification head
        if self.use_estimator:
            self.head = EstimatorQNN()
        else:
            self.head = nn.Sequential(
                nn.Linear(self.fc3.out_features, 1),
                nn.Sigmoid(),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.autoencoder is not None:
            x = self.autoencoder(x)

        logits = self.head(x)
        # Return probabilities for binary classification
        return torch.cat((logits, 1 - logits), dim=-1)

__all__ = [
    "HybridBinaryClassifier",
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "Conv",
    "ConvFilter",
    "EstimatorQNN",
    "EstimatorQNNNet",
]
