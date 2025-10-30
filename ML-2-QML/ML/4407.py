"""Hybrid Quanvolution architecture combining classical convolution, quantum kernel, QCNN, and autoencoder."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class QuanvolutionHybridFilter(nn.Module):
    """Classical filter that mimics a quantum kernel by applying a linear transform
    derived from a pre‑computed quantum kernel matrix.

    The filter first extracts 2×2 patches, applies a single‑channel convolution,
    flattens the result, and then multiplies by the quantum kernel matrix.
    This reproduces the behaviour of a quanvolution layer while remaining fully
    classical and differentiable.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 quantum_kernel: Optional[np.ndarray] = None,
                 threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=False)
        self.quantum_kernel = quantum_kernel
        if quantum_kernel is None:
            # Default to an identity kernel if none provided
            self.quantum_kernel = np.eye(kernel_size * kernel_size)

        # Convert kernel matrix to a learnable weight for back‑prop
        self.kernel_weight = nn.Parameter(
            torch.tensor(self.quantum_kernel, dtype=torch.float32))

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, H, W)
        patches = x.unfold(2, self.kernel_size, self.kernel_size).unfold(3, self.kernel_size, self.kernel_size)
        # patches shape: (B, 1, H', W', k, k)
        patches = patches.contiguous().view(x.size(0), -1, self.kernel_size, self.kernel_size)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._extract_patches(x)          # (B, N, k, k)
        conv_out = self.conv(patches)               # (B, N, 1, 1)
        conv_out = conv_out.view(x.size(0), -1)     # (B, N)
        # Apply quantum kernel: matrix multiplication with kernel_weight
        features = torch.matmul(conv_out, self.kernel_weight)  # (B, k*k)
        return features

class QCNNModel(nn.Module):
    """Adapted QCNN that accepts an arbitrary input dimension."""
    def __init__(self, input_dim: int = 32) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 2 * input_dim), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(2 * input_dim, 2 * input_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(2 * input_dim, input_dim), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(input_dim // 2, input_dim // 2), nn.Tanh())
        self.head = nn.Linear(input_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class AutoencoderNet(nn.Module):
    """Simple MLP autoencoder used as a classical surrogate for a quantum autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

class QuanvolutionHybridClassifier(nn.Module):
    """End‑to‑end hybrid classifier that chains the filter, autoencoder and QCNN."""
    def __init__(self,
                 kernel_size: int = 2,
                 num_classes: int = 10,
                 latent_dim: int = 32,
                 quantum_kernel: Optional[np.ndarray] = None) -> None:
        super().__init__()
        self.filter = QuanvolutionHybridFilter(kernel_size=kernel_size,
                                               quantum_kernel=quantum_kernel)
        # The number of patches per image: (28 - k + 1)^2
        n_patches = (28 - kernel_size + 1) ** 2
        input_dim = kernel_size * kernel_size * n_patches
        self.autoencoder = AutoencoderNet(input_dim=input_dim,
                                          latent_dim=latent_dim)
        self.qcnn = QCNNModel(input_dim=latent_dim)
        self.classifier = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)  # (B, n_patches * k*k)
        # Flatten for autoencoder
        features = features.view(x.size(0), -1)
        latents = self.autoencoder.encode(features)
        qcnn_out = self.qcnn(latents)
        logits = self.classifier(qcnn_out)
        return F.log_softmax(logits, dim=-1)

__all__ = [
    "QuanvolutionHybridFilter",
    "QCNNModel",
    "AutoencoderNet",
    "QuanvolutionHybridClassifier",
]
