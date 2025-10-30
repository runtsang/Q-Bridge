"""Hybrid quantum-classical classifier integrating convolution, auto‑encoding and a variational feed‑forward network."""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

# ----------------------------------------------------------------------
# Classical pre‑processing primitives
# ----------------------------------------------------------------------
class ConvFilter(nn.Module):
    """Convolutional filter emulating a simple quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor | list[list[float]]) -> torch.Tensor:
        """Apply the convolution and return a flattened feature vector."""
        if isinstance(data, list):
            data = torch.tensor(data, dtype=torch.float32)
        data = data.unsqueeze(0).unsqueeze(0)  # shape (1,1,k,k)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.view(-1)  # shape (k*k,)

# ----------------------------------------------------------------------
# Auto‑encoder definition
# ----------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.decode(self.encode(x))

# ----------------------------------------------------------------------
# Hybrid classifier
# ----------------------------------------------------------------------
class QuantumClassifierModel(nn.Module):
    """
    A hybrid model that first extracts local patterns via a classical
    convolution filter, reduces dimensionality with a lightweight
    auto‑encoder, and finally classifies using a variational feed‑forward
    network.  The architecture mirrors the quantum helper, enabling
    seamless switching between classical and quantum back‑ends.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (channels, height, width)
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        classifier_depth: int = 3,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel, threshold=conv_threshold)
        ae_cfg = AutoencoderConfig(
            input_dim=conv_kernel * conv_kernel,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.autoencoder = AutoencoderNet(ae_cfg)
        self.classifier = self._build_classifier(latent_dim, classifier_depth)

    def _build_classifier(self, num_features: int, depth: int) -> nn.Module:
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of images with shape ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, 2)``.
        """
        batch_size = x.shape[0]
        conv_feats = torch.stack(
            [self.conv.run(x[i, 0].cpu().numpy()) for i in range(batch_size)],
            dim=0,
        ).to(x.device)
        encoded = self.autoencoder.encode(conv_feats)
        logits = self.classifier(encoded)
        return logits

    def get_classifier_params(self) -> list[torch.nn.parameter.Parameter]:
        """Return the parameters of the variational classifier."""
        return list(self.classifier.parameters())

__all__ = ["QuantumClassifierModel", "AutoencoderConfig", "AutoencoderNet", "ConvFilter"]
