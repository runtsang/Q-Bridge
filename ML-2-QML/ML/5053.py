from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Iterable, List

# Autoencoder configuration and network (adapted from Autoencoder.py)
class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
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

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> AutoencoderNet:
    config = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(config)

# Classical self‑attention module (adapted from SelfAttention.py)
class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_matrix = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.key_matrix   = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = torch.matmul(inputs, self.query_matrix)
        key   = torch.matmul(inputs, self.key_matrix)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, inputs)

# Classical classifier builder (adapted from QuantumClassifierModel.py)
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    layers: List[nn.Module] = []
    in_dim = num_features
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, list(range(num_features)), weight_sizes, observables

# Hybrid model that stitches auto‑encoder → self‑attention → classifier
class HybridQuantumClassifier(nn.Module):
    """
    A hybrid architecture that processes data through a classical auto‑encoder,
    a self‑attention block, and a feed‑forward classifier.  The structure
    mirrors the quantum helper interface so that the same API can be
    swapped for the quantum implementation in the QML module.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        classifier_depth: int = 3,
        attention_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        embed_dim = attention_embed_dim or latent_dim
        self.attention = SelfAttention(embed_dim)
        self.classifier, _, _, _ = build_classifier_circuit(num_features=embed_dim, depth=classifier_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.autoencoder.encode(x)
        attended = self.attention(latent)
        logits = self.classifier(attended)
        return logits

__all__ = ["HybridQuantumClassifier", "Autoencoder", "AutoencoderConfig", "SelfAttention", "build_classifier_circuit"]
