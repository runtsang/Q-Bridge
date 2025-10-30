"""Hybrid classical regression module with denoising and dropout.

This module extends the original seed by adding an optional denoising autoencoder,
dropout layers, and a simple early‑stopping helper. The goal is to preprocess
noisy superposition data before feeding it to the main regression network.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple, Iterable, Optional

def generate_superposition_data(
    num_features: int,
    samples: int,
    noise_std: float = 0.0,
    *,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate superposition data with optional Gaussian noise.

    Parameters
    ----------
    num_features : int
        Number of input features per sample.
    samples : int
        Number of samples to generate.
    noise_std : float, default 0.0
        Standard deviation of additive Gaussian noise added to the features.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    x : np.ndarray of shape (samples, num_features)
        Input features.
    y : np.ndarray of shape (samples,)
        Target values.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)

    if noise_std > 0.0:
        noise = rng.normal(scale=noise_std, size=x.shape).astype(np.float32)
        x += noise

    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping noisy superposition data."""

    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_std=noise_std
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class DenoisingAutoencoder(nn.Module):
    """Simple autoencoder used for optional denoising."""

    def __init__(self, input_dim: int, latent_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)

class QModel(nn.Module):
    """Main regression network with optional denoising and dropout."""

    def __init__(
        self,
        num_features: int,
        denoise: bool = False,
        dropout: float = 0.1,
        latent_dim: int = 16,
    ):
        super().__init__()
        self.denoise = denoise
        if denoise:
            self.autoencoder = DenoisingAutoencoder(num_features, latent_dim, dropout)
            input_dim = latent_dim
        else:
            self.autoencoder = None
            input_dim = num_features

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.denoise:
            state_batch = self.autoencoder(state_batch)
        return self.net(state_batch).squeeze(-1)

    def train_with_early_stopping(
        self,
        train_loader: Iterable,
        val_loader: Iterable,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int = 100,
        patience: int = 10,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[float, torch.Tensor]:
        """Simple training loop with early stopping based on validation loss.

        Parameters
        ----------
        train_loader : iterable
            DataLoader for training data.
        val_loader : iterable
            DataLoader for validation data.
        criterion : nn.Module
            Loss function.
        optimizer : torch.optim.Optimizer
            Optimizer.
        epochs : int, default 100
            Maximum number of epochs.
        patience : int, default 10
            Number of epochs with no improvement after which training stops.
        device : torch.device, default cpu
            Device to run training on.

        Returns
        -------
        best_val_loss : float
            Minimum validation loss achieved.
        best_state_dict : torch.Tensor
            State dictionary of the best model.
        """
        self.to(device)
        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                states = batch["states"].to(device)
                targets = batch["target"].to(device)
                optimizer.zero_grad()
                outputs = self.forward(states)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Validation
            self.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    states = batch["states"].to(device)
                    targets = batch["target"].to(device)
                    outputs = self.forward(states)
                    val_losses.append(criterion(outputs, targets).item())
            val_loss = np.mean(val_losses)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        if best_state is not None:
            self.load_state_dict(best_state)
        return best_loss, best_state

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
