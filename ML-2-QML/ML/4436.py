"""Hybrid kernel method combining classical RBF, self‑attention, autoencoder and
noise‑aware estimation.

The module mirrors the interface of the original ``QuantumKernelMethod`` but
extends it with:
* a classical self‑attention transformer,
* a lightweight fully‑connected autoencoder,
* a noise‑aware estimator that injects Gaussian shot noise.

The public class :class:`HybridKernelMethod` can be used interchangeably
with the original ``Kernel`` for computing Gram matrices, but it also
provides ``kernel_matrix_with_noise`` for noisy evaluation and
``encode`` for feature extraction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from.SelfAttention import SelfAttention
from.Autoencoder import Autoencoder, AutoencoderConfig
from.FastBaseEstimator import FastEstimator

# --------------------------------------------------------------------------- #
# Classical RBF kernel (from the original seed)
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Gaussian radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Thin wrapper that keeps the original API."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two collections of tensors."""
    k = Kernel(gamma)
    return np.array([[k(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Hybrid kernel implementation
# --------------------------------------------------------------------------- #
class HybridKernelMethod:
    """Hybrid classical‑quantum kernel that uses self‑attention, autoencoding
    and a noise‑aware estimator.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        attention_embed_dim: int = 4,
        autoencoder_cfg: AutoencoderConfig | None = None,
        noise_shots: int | None = None,
        noise_seed: int | None = None,
    ) -> None:
        self.kernel = Kernel(gamma)
        self.attention = SelfAttention()  # returns a ClassicalSelfAttention instance
        cfg = autoencoder_cfg or AutoencoderConfig(
            input_dim=attention_embed_dim,
            latent_dim=32,
            hidden_dims=(128, 64),
            dropout=0.1,
        )
        self.autoencoder = Autoencoder(
            input_dim=cfg.input_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        )
        self.estimator = FastEstimator(self.autoencoder)
        self.noise_shots = noise_shots
        self.noise_seed = noise_seed

    # --------------------------------------------------------------------- #
    # Feature extraction helpers
    # --------------------------------------------------------------------- #
    def _attention_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply classical self‑attention to a single data point."""
        return self.attention.run(
            rotation_params=np.random.randn(self.attention.embed_dim),
            entangle_params=np.random.randn(self.attention.embed_dim),
            inputs=x,
        )

    def encode(self, X: np.ndarray) -> torch.Tensor:
        """Encode a batch of data points into the latent space."""
        X = torch.as_tensor(X, dtype=torch.float32)
        X = self._attention_transform(X.numpy())
        X = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.autoencoder.encode(X)

    # --------------------------------------------------------------------- #
    # Kernel evaluation
    # --------------------------------------------------------------------- #
    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix between two datasets."""
        X_enc = self.encode(X)
        Y_enc = self.encode(Y)
        return np.array([[self.kernel(x, y).item() for y in Y_enc] for x in X_enc])

    # --------------------------------------------------------------------- #
    # Noisy estimation
    # --------------------------------------------------------------------- #
    def kernel_matrix_with_noise(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        shots: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Return a noisy estimate of the Gram matrix using Gaussian shot noise."""
        raw = self.kernel_matrix(X, Y)
        shots = shots or self.noise_shots
        if shots is None:
            return raw
        rng = np.random.default_rng(seed or self.noise_seed)
        noisy = rng.normal(raw, scale=1.0 / np.sqrt(shots))
        return noisy

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "HybridKernelMethod",
]
