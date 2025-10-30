import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

def generate_superposition_data(num_features: int, samples: int):
    """
    Generate synthetic regression data by summing random features and applying a trigonometric target function.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(data.Dataset):
    """
    Classic PyTorch dataset that returns input features and regression targets.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "inputs": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumInspiredLayer(nn.Module):
    """
    A lightweight quantum-inspired layer that applies trainable sine/cosine transformations
    to emulate parameterized quantum rotations in a fully classical setting.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(num_features))
        self.phi = nn.Parameter(torch.randn(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.theta) * x + torch.cos(self.phi) * x

class HybridRegressionModel(nn.Module):
    """
    Hybrid classical‑quantum regression model:
    - Encoder: classical 2‑layer MLP (mirrors EstimatorQNN architecture)
    - Quantum layer: parameterized rotations (QuantumInspiredLayer)
    - Head: final linear output
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 8),
            nn.Tanh()
        )
        self.quantum_layer = QuantumInspiredLayer(8)
        self.head = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.quantum_layer(x)
        return self.head(x).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
