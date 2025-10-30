"""Hybrid regression model combining classical feature extraction and a quantum-inspired random layer.

This module implements a purely classical neural network that mimics the structure of a quantum regression
network while remaining fully compatible with standard PyTorch backends. The model uses a random
weight matrix to simulate the stochastic nature of quantum gates, followed by a trainable
parameterized layer that learns to map the encoded features to the target. The design
facilitates rapid prototyping and serves as a baseline for comparison against the quantum
implementation in the companion module.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_classical_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data inspired by a superposition of basis states.

    The target is a smooth function of the sum of input features, with added periodic
    perturbations to emulate interference patterns.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset for the hybrid regression task.

    Parameters
    ----------
    samples : int
        Number of samples to generate.
    num_features : int
        Dimensionality of the input feature space.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_classical_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class SamplerNetwork(nn.Module):
    """A lightweight sampler that produces a probability distribution over two outputs.

    The network is intentionally shallow to avoid overfitting while still providing a
    stochastic component that can be leveraged by the hybrid model for data augmentation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4, bias=True),
            nn.Tanh(),
            nn.Linear(4, 2, bias=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

class HybridRegressionModel(nn.Module):
    """Classical hybrid regression model with a quantum-inspired random layer.

    The architecture consists of:
    1. A random linear layer that simulates the randomness of a quantum circuit.
    2. A trainable linear transformation that learns to decode the random features.
    3. A final regression head that outputs a scalar prediction.
    """
    def __init__(self, num_features: int, random_dim: int = 32):
        super().__init__()
        # Random layer: fixed weights, no gradients
        self.random_layer = nn.Linear(num_features, random_dim, bias=False)
        self.random_layer.weight.requires_grad = False
        nn.init.normal_(self.random_layer.weight, mean=0.0, std=0.1)

        # Trainable decoder
        self.decoder = nn.Sequential(
            nn.Linear(random_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Apply random projection
        random_features = self.random_layer(state_batch)
        # Decode to prediction
        return self.decoder(random_features).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "SamplerNetwork", "generate_classical_superposition_data"]
