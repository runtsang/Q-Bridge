"""Classical regression model with transformer-based feature extractor.

This module extends the original seed by adding a transformer encoder that
captures interactions between features, and a configurable feed‑forward head.
The dataset generator now supports additive Gaussian noise and a configurable
signal‑to‑noise ratio, making it suitable for robustness studies.

Usage
-----
>>> from QuantumRegression__gen281 import RegressionModel, RegressionDataset, generate_superposition_data
>>> dataset = RegressionDataset(samples=1000, num_features=10, noise_std=0.05)
>>> model = RegressionModel(num_features=10, d_model=64, nhead=4, num_layers=2)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Tuple

def generate_superposition_data(num_features: int,
                                samples: int,
                                noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset where the target is a smooth
    non‑linear function of the input features.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input vectors.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the target.
        Defaults to 0.0 (noise‑free).

    Returns
    -------
    x : np.ndarray
        Feature matrix of shape (samples, num_features).
    y : np.ndarray
        Target vector of shape (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y = y.astype(np.float32)
    if noise_std > 0.0:
        y += np.random.normal(0.0, noise_std, size=y.shape).astype(np.float32)
    return x, y

class RegressionDataset(Dataset):
    """
    PyTorch dataset for the synthetic regression problem.

    Parameters
    ----------
    samples : int
        Number of samples.
    num_features : int
        Dimensionality of the input vectors.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the target.
    """

    def __init__(self,
                 samples: int,
                 num_features: int,
                 noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_std
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class RegressionModel(nn.Module):
    """
    Transformer‑based regression model.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input vectors.
    d_model : int, default 64
        Hidden dimension of the transformer encoder.
    nhead : int, default 4
        Number of attention heads.
    num_layers : int, default 2
        Number of transformer encoder layers.
    dropout : float, default 0.1
        Dropout probability.
    """

    def __init__(self,
                 num_features: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Input tensor of shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Predicted targets of shape (batch,).
        """
        x = self.input_proj(state_batch).unsqueeze(0)  # (1, batch, d_model)
        x = self.transformer(x)  # (1, batch, d_model)
        x = x.squeeze(0)  # (batch, d_model)
        return self.head(x).squeeze(-1)

__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
