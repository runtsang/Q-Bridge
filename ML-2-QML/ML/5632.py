"""Combined classical regression with autoencoder preprocessing.

This module defines a unified API that pairs a classical autoencoder
with a regression head.  The dataset is constructed from superposition
state angles, mirroring the quantum counterpart.  The `RegressionAutoencoder`
model first encodes the input features into a latent representation
using a configurable autoencoder, then applies a small feed‑forward
regressor to predict the target.  The module also exposes convenient
training utilities that mirror the quantum training loop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Iterable

# Import the classical autoencoder utilities
from Autoencoder import Autoencoder, AutoencoderConfig, train_autoencoder


def generate_superposition_data(
    num_features: int, samples: int, seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset derived from superpositions.
    The input features are uniformly sampled and the target is a smooth
    trigonometric function of their sum.  This mirrors the quantum
    dataset used in the original implementation.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset that yields a dictionary containing the raw features
    and the target label.  The features can later be passed through
    an autoencoder before regression.
    """
    def __init__(self, samples: int, num_features: int, seed: int | None = None):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, seed
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class RegressionAutoencoder(nn.Module):
    """
    A two‑stage model that first compresses the input with a classical
    autoencoder and then predicts a continuous target from the latent
    representation.  The autoencoder is fully configurable through
    `AutoencoderConfig`.
    """
    def __init__(
        self,
        num_features: int,
        *,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        regressor_hidden: int = 64,
    ):
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim=num_features,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, regressor_hidden),
            nn.ReLU(),
            nn.Linear(regressor_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        latent = self.autoencoder.encode(features)
        return self.regressor(latent).squeeze(-1)


def train_regression(
    model: RegressionAutoencoder,
    dataset: Dataset,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Simple training loop for the regression head.  The autoencoder
    is frozen during regression training; only the regressor is updated.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.autoencoder.eval()  # freeze autoencoder

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.regressor.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            states = batch["states"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad()
            preds = model(states)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * states.size(0)
        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "RegressionAutoencoder",
    "RegressionDataset",
    "generate_superposition_data",
    "train_regression",
]
