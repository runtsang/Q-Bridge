"""Hybrid classical regression model that mimics quantum behaviour via a learned fully‑connected block."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Iterable, Tuple, List, Optional

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic nonlinear regression data with optional Gaussian noise."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0:
        y += np.random.randn(samples).astype(np.float32) * noise_std
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapper that returns feature tensors and regression targets."""
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.1):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise_std)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridRegression(nn.Module):
    """Classical regression network that emulates a quantum layer via a learned fully‑connected block."""
    def __init__(self, num_features: int, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64)):
        super().__init__()
        # Classical encoder
        encoder_layers: List[nn.Module] = []
        in_dim = num_features
        for h in hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, h), nn.ReLU(inplace=True)])
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum‑inspired fully‑connected block (from FCL reference)
        self.qfc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.qfc(z).squeeze(-1)
        return out

def train_regression(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None
) -> List[float]:
    """Training loop for the hybrid regression model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        total = 0.0
        model.train()
        for batch in dataloader:
            xs = batch["states"].to(device)
            ys = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xs)
            loss = loss_fn(preds, ys)
            loss.backward()
            optimizer.step()
            total += loss.item() * xs.size(0)
        epoch_loss = total / len(dataloader.dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data", "train_regression"]
