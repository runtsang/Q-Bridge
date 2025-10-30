"""Hybrid regression model with classical encoder and sampler-based approximation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def generate_features_and_target(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic regression data.
    Features are sampled uniformly; labels are a smooth non‑linear function of the
    sum of the features to mimic a quantum superposition.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class SamplerModule(nn.Module):
    """
    Mimics a quantum sampler circuit with a tiny neural network.
    Outputs a probability distribution over two outcomes.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expect inputs shape (..., 2); apply softmax to get probabilities
        return F.softmax(self.net(inputs), dim=-1)


class RegressionDataset(Dataset):
    """
    Dataset that returns both the raw features and a simulated quantum
    probability distribution produced by a small sampler network.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_features_and_target(num_features, samples)
        self.sampler = SamplerModule()

    def __len__(self) -> int:  # type: ignore[override]
        return self.features.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        probs = self.sampler(torch.tensor(self.features[idx], dtype=torch.float32))
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "probs": probs,  # 2‑dimensional probability vector
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridRegression(nn.Module):
    """
    Classical model that learns to map raw features and sampler probabilities
    to a regression target. The head is a simple linear layer.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.head = nn.Linear(18, 1)  # 16 from encoder + 2 from sampler probs

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        feat = self.encoder(batch["features"])
        probs = batch["probs"]
        combined = torch.cat([feat, probs], dim=-1)
        return self.head(combined).squeeze(-1)


__all__ = ["HybridRegression", "RegressionDataset", "SamplerModule", "generate_features_and_target"]
