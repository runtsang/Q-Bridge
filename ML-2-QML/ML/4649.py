"""Classical hybrid binary classifier combining CNN, autoencoder, and RBF kernel embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Autoencoder definitions (simplified from the original Autoencoder module)
class AutoencoderConfig:
    """Configuration for the fully‑connected autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with symmetric encoder/decoder."""
    def __init__(self, config: AutoencoderConfig):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# RBF kernel layer using trainable prototypes
class RBFKernelLayer(nn.Module):
    """Computes an RBF kernel between latent vectors and learnable prototypes."""
    def __init__(self, latent_dim: int, n_prototypes: int = 10, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        # Prototypes are learnable parameters
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, latent_dim)
        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (batch, n_prototypes, latent_dim)
        dist_sq = (diff ** 2).sum(dim=-1)
        return torch.exp(-self.gamma * dist_sq)

# Hybrid binary classifier
class HybridBinaryClassifier(nn.Module):
    """CNN backbone + autoencoder + kernel embedding + linear head."""
    def __init__(self,
                 num_classes: int = 2,
                 latent_dim: int = 32,
                 n_prototypes: int = 10,
                 gamma: float = 1.0):
        super().__init__()
        # Convolutional feature extractor (same as the original QCNet without the final fc layers)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Autoencoder to compress the flattened convolutional feature map
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(input_dim=55815, latent_dim=latent_dim)
        )

        # Kernel embedding
        self.kernel = RBFKernelLayer(latent_dim=latent_dim,
                                     n_prototypes=n_prototypes,
                                     gamma=gamma)

        # Linear classifier from kernel space to logits
        self.classifier = nn.Linear(n_prototypes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop2(x)
        x = torch.flatten(x, 1)          # (batch, 55815)

        # Encode to latent space
        latent = self.autoencoder.encode(x)   # (batch, latent_dim)

        # Kernel embedding
        k = self.kernel(latent)               # (batch, n_prototypes)

        # Linear head
        logits = self.classifier(k)           # (batch, num_classes)
        probs = torch.softmax(logits, dim=-1)
        return probs

__all__ = ["AutoencoderConfig", "AutoencoderNet", "RBFKernelLayer",
           "HybridBinaryClassifier"]
