"""Hybrid classical classifier that emulates the quantum workflow.

The class combines an autoencoder feature extractor with a small
feed‑forward head inspired by the EstimatorQNN example.  It is a
drop‑in replacement for the original build_classifier_circuit factory
while providing a fully classical training pipeline.

Architecture
------------
* Autoencoder → latent representation
* Linear expansion to match the number of qubits in the quantum model
* Classification head (EstimatorQNN style)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ----------------- Autoencoder (reference 3) -----------------
class AutoencoderConfig:
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


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

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


# ----------------- EstimatorQNN (reference 2) -----------------
def EstimatorQNN(input_dim: int = 4) -> nn.Module:
    """Return a tiny feed‑forward network that can serve as a classification head."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)

    return EstimatorNN()


# ----------------- Hybrid classifier -----------------
class HybridQuantumClassifier(nn.Module):
    """
    Classical hybrid model that emulates the structure of the quantum
    classifier from the original seed.

    Architecture:
        * Autoencoder → latent representation
        * Linear expansion to match the number of qubits in the quantum model
        * Small feed‑forward head (EstimatorQNN style)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        num_qubits: int = 4,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.num_qubits = num_qubits
        self.depth = depth

        # Expand latent vector to match the quantum circuit width
        self.expand = nn.Linear(latent_dim, num_qubits)

        # Classification head inspired by EstimatorQNN
        self.head = EstimatorQNN(num_qubits)

    def forward(self, x: Tensor) -> Tensor:
        z = self.autoencoder.encode(x)
        z_expanded = self.expand(z)
        logits = self.head(z_expanded)
        return logits


__all__ = [
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "EstimatorQNN",
    "HybridQuantumClassifier",
]
