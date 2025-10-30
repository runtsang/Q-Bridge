"""Merging a classical autoencoder with a regression head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate sinusoidal labels from a random superposition of features."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset returning features and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.x, self.y = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.x[idx], dtype=torch.float32),
            "target": torch.tensor(self.y[idx], dtype=torch.float32),
        }


# ---------- Classical Autoencoder ----------
class AutoencoderConfig:
    """Configuration for the autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder."""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class HybridRegressionAutoencoder(nn.Module):
    """Regression head on top of an autoencoder."""
    def __init__(self, num_features: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        cfg = AutoencoderConfig(num_features, latent_dim, hidden_dims, dropout)
        self.autoencoder = AutoencoderNet(cfg)
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        return self.regressor(z).squeeze(-1)


def train_hybrid(
    model: HybridRegressionAutoencoder,
    dataset: Dataset,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Train the autoencoder and regression head jointly."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    history = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            states = batch["states"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            pred = model(states)
            loss = mse(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * states.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "HybridRegressionAutoencoder",
    "RegressionDataset",
    "generate_superposition_data",
    "train_hybrid",
]
