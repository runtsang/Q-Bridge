"""Hybrid classical classifier combining an autoencoder feature extractor and a feed‑forward head."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class AutoencoderNet(nn.Module):
    """Lightweight MLP autoencoder with configurable depth."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int,...] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


class HybridClassifier(nn.Module):
    """Classifier that first compresses inputs via an autoencoder then classifies."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int,...] = (128, 64),
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(input_dim, latent_dim, hidden_dims)
        layers = [nn.Linear(latent_dim, hidden_dims[0]), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dims[0], hidden_dims[0]), nn.ReLU()])
        layers.append(nn.Linear(hidden_dims[0], 2))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        z = self.autoencoder.encode(x)
        return self.classifier(z)


def build_classifier_circuit(
    num_features: int,
    depth: int,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Return a hybrid classifier network and metadata mimicking the quantum helper.
    """
    model = HybridClassifier(num_features, latent_dim, hidden_dims, depth)
    weight_sizes = [p.numel() for p in model.parameters()]
    encoding = list(range(num_features))
    observables = list(range(2))
    return model, encoding, weight_sizes, observables


__all__ = ["AutoencoderNet", "HybridClassifier", "build_classifier_circuit"]
