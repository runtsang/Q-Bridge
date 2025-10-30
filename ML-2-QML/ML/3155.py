"""Hybrid classical autoencoding regression model.

This module defines a dataset that generates superposition states,
an autoencoder that compresses the raw features,
and a regression network that operates on the latent representation.
The design combines the ideas from the original quantum regression
and the classical autoencoder, offering a scalable hybrid pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Data generation
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset of raw features and targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# Autoencoder
class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# Regression head
class RegressionHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# Hybrid pipeline
class HybridRegressionAutoencoder:
    """End‑to‑end model that first compresses data with an autoencoder
    and then predicts a continuous target.
    """
    def __init__(self, num_features: int, latent_dim: int = 32, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = AutoencoderNet(AutoencoderConfig(num_features, latent_dim)).to(self.device)
        self.regressor = RegressionHead(latent_dim).to(self.device)

    def train_autoencoder(self, data: torch.Tensor, epochs: int = 50, lr: float = 1e-3) -> list[float]:
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch[0].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                recon = self.autoencoder(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

    def train_regression(self, dataset: Dataset, epochs: int = 100, lr: float = 1e-3) -> list[float]:
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(list(self.autoencoder.parameters()) + list(self.regressor.parameters()), lr=lr)
        loss_fn = nn.MSELoss()
        history = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                features = batch["features"].to(self.device)
                target = batch["target"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                latent = self.autoencoder.encode(features)
                pred = self.regressor(latent)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * features.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        with torch.no_grad():
            latent = self.autoencoder.encode(X)
            return self.regressor(latent)

__all__ = ["HybridRegressionAutoencoder", "RegressionDataset", "AutoencoderNet", "AutoencoderConfig", "RegressionHead"]
