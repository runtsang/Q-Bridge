"""Hybrid classical regression model that mirrors the quantum encoder structure.

The network is constructed from a shallow encoder that mimics the quantum
parameter‑encoding layer, followed by a configurable feed‑forward backbone.
This mirrors the quantum architecture so that the classical counterpart can
serve as a baseline and as a training scaffold for the quantum model.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic data with a sinusoidal target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns feature vectors and regression targets."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """Return a classical network that mimics the quantum encoder+ansatz structure.

    The returned tuple contains:
        - the nn.Sequential model,
        - a list of encoding indices (the input feature indices),
        - a list of weight sizes for each linear layer,
        - a list of output feature indices (the regression head input).
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 1)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(1))  # regression output
    return network, encoding, weight_sizes, observables


class HybridRegression(nn.Module):
    """Classical regression model with a quantum‑style encoder."""

    def __init__(self, num_features: int, hidden_dim: int = 64, depth: int = 2, use_encoder: bool = True):
        super().__init__()
        self.use_encoder = use_encoder
        if use_encoder:
            # Simple linear encoder that imitates a parameter‑encoding layer
            self.encoder = nn.Linear(num_features, hidden_dim)
        else:
            self.encoder = nn.Identity()

        # Feed‑forward backbone inspired by the quantum ansatz
        layers = []
        in_dim = hidden_dim
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
        self.backbone = nn.Sequential(*layers)

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.encoder(state_batch)
        x = self.backbone(x)
        return self.head(x).squeeze(-1)


__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data", "build_classifier_circuit"]
