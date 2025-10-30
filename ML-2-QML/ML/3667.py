"""Hybrid Quanvolution–Autoencoder combining classical convolution and dense autoencoder architecture.

The class wraps a patch‑wise quantum‑kernel inspired linear transformation
followed by a fully‑connected encoder/decoder.  The quantum kernel is
implemented as a random orthogonal matrix generated once and reused for
all patches, mimicking the action of a shallow quantum circuit while
remaining fully classical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Tuple

class RandomOrthogonalLayer(nn.Module):
    """Apply a fixed random orthogonal transform to each patch."""
    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        rng = np.random.default_rng(seed)
        mat = rng.standard_normal((out_features, in_features))
        q, _ = np.linalg.qr(mat, mode="reduced")
        self.register_buffer("weight", torch.from_numpy(q.astype(np.float32)))
        self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class QuanvolutionAutoencoder(nn.Module):
    """Hybrid autoencoder that first extracts 2×2 patches with a quantum‑kernel
    inspired linear layer and then compresses the resulting features
    using a dense encoder/decoder stack.
    """
    def __init__(
        self,
        *,
        in_channels: int = 1,
        patch_size: int = 2,
        stride: int = 2,
        quantum_out_dim: int = 4,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.quantum_out_dim = quantum_out_dim

        # Classical “quanvolution” stage
        self.conv = nn.Conv2d(
            in_channels, quantum_out_dim, kernel_size=patch_size, stride=stride
        )
        self.ortho = RandomOrthogonalLayer(
            in_features=quantum_out_dim, out_features=quantum_out_dim
        )

        # Compute number of patches
        dummy = torch.zeros(1, in_channels, 28, 28)
        num_patches = ((28 - patch_size) // stride + 1) ** 2
        flat_dim = num_patches * quantum_out_dim

        # Autoencoder backbone
        encoder_layers = []
        in_dim = flat_dim
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
        decoder_layers.append(nn.Linear(in_dim, flat_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical patch extraction
        patches = self.conv(x)  # shape (B, quantum_out_dim, H', W')
        B = patches.size(0)
        patches = patches.view(B, self.quantum_out_dim, -1).transpose(1, 2)
        # Apply orthogonal transform
        patches = self.ortho(patches)  # (B, num_patches, quantum_out_dim)
        patches = patches.reshape(B, -1)  # flatten

        # Encode & decode
        latent = self.encoder(patches)
        recon = self.decoder(latent)

        # Reshape back to image form
        recon = recon.view(B, self.quantum_out_dim, -1).transpose(1, 2)
        recon = recon.view(B, -1, *self.conv.weight.shape[2:])  # approximate inverse conv
        return recon

__all__ = ["QuanvolutionAutoencoder", "RandomOrthogonalLayer"]
