"""
HybridQCNN: Classical backbone with optional auto‑encoding.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from dataclasses import dataclass
from typing import Tuple, Iterable, Any, Dict

# --------------------------------------------------------------------------- #
# Data generation utilities
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    Returns input features and a regression target.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    features = np.zeros((samples, num_features), dtype=np.float32)
    for i in range(samples):
        # Encode theta and phi linearly for simplicity
        features[i, 0] = np.cos(thetas[i])
        features[i, 1] = np.sin(thetas[i])
        features[i, 2] = np.cos(phis[i])
        features[i, 3] = np.sin(phis[i])
        if num_features > 4:
            features[i, 4:] = np.random.normal(size=num_features - 4)
    targets = np.sin(2 * thetas) * np.cos(phis)
    return features, targets.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns a state vector and a scalar target."""
    def __init__(self, samples: int, num_features: int):
        self.states, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Auto‑encoding helper
# --------------------------------------------------------------------------- #

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder with configurable depth."""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        enc_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(cfg)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(TensorDataset(_as_tensor(data)), batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)
    return history


# --------------------------------------------------------------------------- #
# Hybrid classical backbone
# --------------------------------------------------------------------------- #

class HybridQCNNModel(nn.Module):
    """
    Convolution‑style fully‑connected network that optionally accepts
    a latent vector from an auto‑encoder before the final regression head.
    """
    def __init__(self, input_dim: int, latent_dim: int = 0, hidden_dims: Tuple[int,...] = (64, 32, 16)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        if latent_dim:
            # Merge latent representation before the head
            layers.append(nn.Linear(in_dim + latent_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        if latent is not None:
            x = torch.cat([x, latent], dim=-1)
        out = self.net(x)
        return torch.sigmoid(out.squeeze(-1))


def QCNN() -> HybridQCNNModel:
    """
    Factory returning a default HybridQCNNModel instance.
    """
    return HybridQCNNModel(input_dim=8, latent_dim=0)


__all__ = ["QCNN", "HybridQCNNModel", "RegressionDataset", "Autoencoder", "train_autoencoder", "generate_superposition_data"]
