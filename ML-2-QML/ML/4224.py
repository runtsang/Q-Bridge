import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset that mirrors the quantum superposition
    generator from the QML seed.  The function is intentionally generic so it can be
    used by both the classical and quantum modules."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that yields a dict of tensors for compatibility with the ML and QML APIs."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class ClassicalRegressor(nn.Module):
    """A lightweight MLP that can be fine‑tuned after the quantum encoder."""
    def __init__(self, num_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class QuantumEncoder(nn.Module):
    """Quantum circuit that encodes the classical input and returns expectation values."""
    def __init__(self, num_wires: int, layer_depth: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.layer_depth = layer_depth
        self.enc = nn.ModuleList()
        for _ in range(layer_depth):
            self.enc.append(nn.Linear(num_wires, num_wires))  # placeholder for layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mocked forward that returns a tensor of size (batch, num_wires)
        return torch.randn(x.size(0), self.num_wires, device=x.device)

class UnifiedRegressor(nn.Module):
    """A joint model that integrates quantum and classical components."""
    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.quantum = QuantumEncoder(num_wires)
        self.classical = ClassicalRegressor(num_features)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qfeat = self.quantum(x)
        cfeat = self.classical(x)
        combined = qfeat + cfeat.unsqueeze(1)  # broadcast add
        return self.head(combined).squeeze(-1)

__all__ = ["UnifiedRegressor", "RegressionDataset", "generate_superposition_data"]
