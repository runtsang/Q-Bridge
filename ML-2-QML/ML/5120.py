"""Hybrid classical neural network combining quanvolution, RBF kernel, autoencoder, and regressor."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuanvolutionFilter(nn.Module):
    """Standard 2‑D convolution that emulates the original quanvolution idea."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class RBFKernel(nn.Module):
    """Radial‑basis‑function kernel usable as a drop‑in replacement for a quantum kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * (diff * diff).sum(-1, keepdim=True))


class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder mirroring the reference implementation."""
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: tuple[int, int] = (128, 64)) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


class EstimatorNN(nn.Module):
    """Simple feed‑forward regressor used as the final head."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * 14 * 14, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class QuanvolutionAutoencoderQNN(nn.Module):
    """
    Composite model that chains a classical quanvolution filter, an RBF kernel,
    an autoencoder, and a regression head.  The API mirrors the quantum
    counterpart so that downstream experiments can swap implementations
    without altering the training loop.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.kernel = RBFKernel(gamma)
        self.autoencoder = AutoencoderNet(
            input_dim=4 * 14 * 14,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        self.estimator = EstimatorNN()
        self._training_features: torch.Tensor | None = None

    def fit_features(self, features: torch.Tensor) -> None:
        """Store a reference set of features for kernel evaluation."""
        self._training_features = features.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        qfeat = self.qfilter(x)
        if self._training_features is not None:
            # pairwise kernel matrix between input and stored reference set
            kmat = self.kernel(qfeat, self._training_features)
            qfeat = kmat
        ae_out = self.autoencoder(qfeat)
        logits = self.estimator(ae_out)
        return logits


__all__ = ["QuanvolutionAutoencoderQNN"]
