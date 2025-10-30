"""Classical autoencoder-based binary classifier.

This module defines a lightweight autoencoder and a classifier that
uses the latent representation as features.  It mirrors the quantum
version in the QML module but replaces the quantum expectation head
with a dense sigmoid layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with configurable depth."""

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class HybridAutoencoderClassifier(nn.Module):
    """Classifier that uses the autoencoder's latent space as features."""

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(input_dim, latent_dim,
                                          hidden_dims, dropout)
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        logits = self.classifier(z)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["AutoencoderNet", "HybridAutoencoderClassifier"]
