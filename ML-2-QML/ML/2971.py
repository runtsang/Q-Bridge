"""Hybrid regression model combining classical neural network with quantum‑inspired random layer and EstimatorQNN head."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and labels.

    The function reuses the quantum seed but returns real‑valued features
    (absolute values of the complex amplitudes) so that the model can be
    trained entirely on classical tensors.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    # Convert states to real‑valued feature vectors
    real_features = np.abs(states)
    return real_features.astype(np.float32), labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping the superposition states as real feature vectors."""
    def __init__(self, samples: int, num_wires: int):
        self.features, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridRegressionModel(nn.Module):
    """Classical hybrid regressor that mimics the quantum structure.

    - An encoder linear layer projects the input to 32 dimensions.
    - A fixed random linear layer emulates the quantum random circuit.
    - The head uses the same small feed‑forward network as EstimatorQNN.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.encoder = nn.Linear(num_wires, 32)
        # Fixed random layer (non‑trainable)
        self.random_layer = nn.Linear(32, 32, bias=False)
        nn.init.normal_(self.random_layer.weight)
        self.random_layer.weight.requires_grad = False
        # EstimatorQNN‑style head
        self.head = nn.Sequential(
            nn.Linear(32, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.random_layer(x)
        return self.head(x).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
