"""
Hybrid regression module (classical).

Provides:
* A data generator for superposition‑style inputs.
* A lightweight regression dataset.
* A multi‑layer perceptron autoencoder (configurable).
* A regression head that consumes the autoencoder’s latent vectors.
* A fast, noise‑aware estimator for batch evaluation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Iterable, List, Sequence

# ------------------------------------------------------------------
# Data generation utilities
# ------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce a set of real‑valued feature vectors and regression targets
    derived from quantum‑style superpositions.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class RegressionDataset(Dataset):
    """
    PyTorch Dataset that yields feature tensors and target scalars.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ------------------------------------------------------------------
# Autoencoder utilities
# ------------------------------------------------------------------
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class AutoencoderConfig:
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        # Encoder
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

        # Decoder
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

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    return AutoencoderNet(AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout))

# ------------------------------------------------------------------
# Regression head
# ------------------------------------------------------------------
class RegressionHead(nn.Module):
    """
    Simple feed‑forward head that maps latent vectors to a scalar output.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

# ------------------------------------------------------------------
# Hybrid regression model
# ------------------------------------------------------------------
class HybridRegression(nn.Module):
    """
    Combines an autoencoder and a regression head.

    Parameters
    ----------
    input_dim : int
        dimensionality of the raw feature vector.
    latent_dim : int, optional
        dimensionality of the autoencoder bottleneck.
    """
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.autoencoder = Autoencoder(input_dim, latent_dim=latent_dim)
        self.head = RegressionHead(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        return self.head(z)

# ------------------------------------------------------------------
# Training helper
# ------------------------------------------------------------------
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
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# ------------------------------------------------------------------
# Estimator utilities (fast, noise‑aware)
# ------------------------------------------------------------------
class FastBaseEstimator:
    """
    Evaluate a PyTorch model over a batch of parameter sets, optionally adding shot noise.
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(
        self,
        observables: Iterable[callable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[float]]:
        obs = list(observables) or [lambda out: out.mean().item()]
        raw = self._evaluate(obs, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def _evaluate(
        self,
        observables: list[callable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        self.model.eval()
        results: list[list[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inp = _as_tensor(params).unsqueeze(0)
                out = self.model(inp).squeeze(-1)
                row = [float(obs(out)) for obs in observables]
                results.append(row)
        return results

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "Autoencoder",
    "AutoencoderNet",
    "RegressionHead",
    "HybridRegression",
    "train_autoencoder",
    "FastBaseEstimator",
]
