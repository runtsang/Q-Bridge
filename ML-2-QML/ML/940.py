"""Enhanced classical regression model and dataset."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data from the superposition state cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    Adds optional Gaussian noise to labels.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(scale=noise_std, size=y.shape)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset returning a dictionary with'states' (features) and 'target' (labels).
    """
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise_std)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    A deeper neural network with dropout and L2 regularization support.
    """
    def __init__(self, num_features: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

def train_model(
    model: QModel,
    dataset: RegressionDataset,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_grad: float | None = 1.0,
    device: torch.device | str = "cpu",
) -> list[float]:
    """
    Simple training loop with L2 regularization and optional gradient clipping.
    Returns a list of training losses.
    """
    model = model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            states = batch["states"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, targets)
            loss.backward()
            if clip_grad is not None:
                clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            epoch_loss += loss.item() * states.size(0)
        epoch_loss /= len(dataloader.dataset)
        losses.append(epoch_loss)
    return losses
