"""Hybrid regression model combining classical neural network with a sampler network.

This module defines:
- generate_superposition_data: produce synthetic regression data with sinusoidal target.
- RegressionDataset: torch.utils.data.Dataset.
- SamplerNetwork: a small feed‑forward network that outputs a probability distribution over two outcomes.
- HybridRegression: a fully‑connected network with batch normalization and dropout, optionally accepting a sampler output.

The API matches the original QuantumRegression.py for easy swapping during experiments.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data where target is a nonlinear function of a linear sum of features.

    The function is more expressive than the original seed by adding a higher‑order cosine term
    and a random phase shift, producing a richer training signal for the sampler network.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = np.sum(x, axis=1)
    y = np.sin(angles) + 0.3 * np.cos(1.7 * angles + 0.5)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that mirrors the quantum data generator but returns real tensors."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class SamplerNetwork(nn.Module):
    """Classical sampler network that emulates a 2‑outcome probability distribution."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return a probability distribution over two classes."""
        return F.softmax(self.net(inputs), dim=-1)

class HybridRegression(nn.Module):
    """Classical regression network with optional sampler integration."""
    def __init__(self, num_features: int, use_sampler: bool = False):
        super().__init__()
        self.use_sampler = use_sampler
        # Main regression head
        self.main = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
        # Optional sampler branch
        self.sampler = SamplerNetwork() if use_sampler else None

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass. If a sampler is attached, its output is concatenated before the head."""
        x = state_batch
        if self.use_sampler:
            # Use a simple 2‑dimensional embedding of the state for sampling
            embed = torch.mean(state_batch, dim=1, keepdim=True).repeat(1, 2)
            sample_probs = self.sampler(embed)
            x = torch.cat([x, sample_probs], dim=1)
        return self.main(x).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
