"""Hybrid classical regression model combining CNN feature extraction and a linear head.

This module mirrors the structure of the original QuantumRegression seed but augments
the classical branch with convolutional layers inspired by the Quantum‑NAT example.
The dataset generation and target labels remain identical to the original seed, ensuring
compatibility with existing evaluation scripts.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    The target is a smooth nonlinear function of the underlying angles.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset that returns a dictionary with the raw state vector and the target.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridRegressionModel(nn.Module):
    """
    Classical CNN + fully‑connected head for regression.
    The architecture is inspired by the Quantum‑NAT CNN and the original regression model.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.input_dim = 2 ** num_wires

        # Feature extractor: 1‑D convolutional pipeline
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Compute flattened feature size dynamically
        dummy = torch.zeros(1, 1, self.input_dim)
        dummy_out = self.features(dummy)
        flat_size = dummy_out.view(1, -1).size(1)

        # Fully‑connected regression head
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (batch_size, 2**num_wires)

        Returns
        -------
        torch.Tensor
            Predicted scalar values, shape (batch_size,)
        """
        x = state_batch.unsqueeze(1)  # (bsz, 1, input_dim)
        feats = self.features(x)
        flattened = feats.view(feats.size(0), -1)
        out = self.fc(flattened)
        return self.norm(out).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
