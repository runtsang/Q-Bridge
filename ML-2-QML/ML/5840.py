"""
Hybrid classical autoencoder with integrated classifier.
Provides a unified interface that mirrors the quantum helper while
offering a richer neural‑network backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn

@dataclass
class HybridConfig:
    """Configuration for the hybrid autoencoder‑classifier."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    classifier_depth: int = 2  # number of hidden layers in the classifier


class HybridAutoencoderClassifier(nn.Module):
    """A light‑weight MLP autoencoder with a binary classifier head."""
    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Classifier
        clf_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for _ in range(cfg.classifier_depth):
            clf_layers.append(nn.Linear(in_dim, cfg.latent_dim))
            clf_layers.append(nn.ReLU())
            in_dim = cfg.latent_dim
        clf_layers.append(nn.Linear(in_dim, 2))
        self.classifier = nn.Sequential(*clf_layers)

        # Decoder
        dec_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        logits = self.classifier(z)
        recon = self.decode(z)
        return logits, recon

    def weight_sizes(self) -> List[int]:
        """Return the number of trainable parameters per sub‑module."""
        sizes: List[int] = []
        for m in [self.encoder, self.classifier, self.decoder]:
            for p in m.parameters():
                sizes.append(p.numel())
        return sizes


def HybridAutoencoderClassifierFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    classifier_depth: int = 2,
) -> HybridAutoencoderClassifier:
    """Convenience factory mirroring the quantum helper."""
    cfg = HybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        classifier_depth=classifier_depth,
    )
    return HybridAutoencoderClassifier(cfg)


def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier and metadata similar to the quantum variant.

    Returns:
        network: the PyTorch model
        encoding: indices of input features
        weight_sizes: number of parameters per layer
        observables: dummy observables for API consistency
    """
    cfg = HybridConfig(
        input_dim=num_features,
        classifier_depth=depth,
    )
    network = HybridAutoencoderClassifierFactory(
        input_dim=num_features,
        classifier_depth=depth,
    )
    encoding = list(range(num_features))
    weight_sizes = network.weight_sizes()
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


__all__ = [
    "HybridAutoencoderClassifier",
    "HybridAutoencoderClassifierFactory",
    "build_classifier_circuit",
    "HybridConfig",
]
