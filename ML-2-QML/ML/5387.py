"""Hybrid regression module with optional autoencoder and transformer.

This module extends the original QuantumRegression seeds by adding:
- A flexible data generator that can produce classical feature vectors or
  complex amplitude arrays for quantum states.
- A lightweight MLP that can be swapped with a quantum encoder.
- An optional classical autoencoder that compresses the input before
  regression, inspired by the Autoencoder seed.
- A simple transformer encoder that can be used as a pre‑processing stage,
  borrowing the design of QTransformerTorch.

The public API remains unchanged: RegressionDataset, QModel, and
generate_superposition_data are still available for backward
compatibility.  New convenience classes are also exported.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int,
                                samples: int,
                                *,
                                use_complex: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a tuple (features, labels).  The features are either a
    real-valued matrix of shape (samples, num_features) or a
    complex-valued matrix of shape (samples, 2**num_features) when
    ``use_complex`` is True.  Labels are computed as
    sin(θ) + 0.1*cos(2θ) where θ = sum of the features.
    """
    if use_complex:
        omega_0 = np.zeros(2**num_features, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2**num_features, dtype=complex)
        omega_1[-1] = 1.0

        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)

        states = np.zeros((samples, 2**num_features), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1

        labels = np.sin(2 * thetas) * np.cos(phis)
        return states, labels.astype(np.float32)
    else:
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary with ``states`` and ``target``.
    ``states`` can be real or complex depending on ``use_complex``.
    """
    def __init__(self, samples: int, num_features: int, *, use_complex: bool = False):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, use_complex=use_complex
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        dtype = torch.cfloat if self.features.dtype.kind == 'c' else torch.float32
        return {
            "states": torch.tensor(self.features[index], dtype=dtype),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Classical MLP
# --------------------------------------------------------------------------- #
class ClassicalMLP(nn.Module):
    """Small feed‑forward regressor inspired by EstimatorQNN."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x).squeeze(-1)

# --------------------------------------------------------------------------- #
# Autoencoder
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        encoder = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder.append(nn.Linear(in_dim, hidden))
            encoder.append(nn.ReLU())
            if dropout > 0.0:
                encoder.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder.append(nn.Linear(in_dim, hidden))
            decoder.append(nn.ReLU())
            if dropout > 0.0:
                decoder.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
# Transformer encoder (optional)
# --------------------------------------------------------------------------- #
class TransformerEncoder(nn.Module):
    """Simple transformer encoder block."""
    def __init__(self, embed_dim: int, num_heads: int = 4, ffn_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim,
                                       nhead=num_heads,
                                       dim_feedforward=ffn_dim,
                                       dropout=dropout,
                                       activation="relu")
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# --------------------------------------------------------------------------- #
# Hybrid regression model
# --------------------------------------------------------------------------- #
class QModel(nn.Module):
    """
    Hybrid regression model that can optionally prepend an autoencoder or
    transformer encoder before the regression head.  The choice of encoder
    is controlled by the ``preprocess`` argument.
    """
    def __init__(self,
                 input_dim: int,
                 *,
                 use_autoencoder: bool = False,
                 use_transformer: bool = False,
                 autoencoder_cfg: dict | None = None,
                 transformer_cfg: dict | None = None):
        super().__init__()
        self.preprocess = None
        if use_autoencoder:
            cfg = autoencoder_cfg or {}
            self.preprocess = AutoencoderNet(input_dim, **cfg)
            feature_dim = cfg.get("latent_dim", 32)
        elif use_transformer:
            cfg = transformer_cfg or {}
            self.preprocess = TransformerEncoder(input_dim, **cfg)
            feature_dim = input_dim
        else:
            feature_dim = input_dim

        self.head = ClassicalMLP(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.preprocess is not None:
            x = self.preprocess(x)
        return self.head(x)

# --------------------------------------------------------------------------- #
# Backward‑compatibility aliases
# --------------------------------------------------------------------------- #
# The original seed exposed a class called QModel.  We keep that name
# for backward compatibility while also providing a more descriptive
# alias for advanced usage.
HybridRegressionModel = QModel

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "QModel",
    "HybridRegressionModel",
    "AutoencoderNet",
    "TransformerEncoder",
]
