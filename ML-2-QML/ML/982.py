"""Hybrid classical–quantum regression model with joint training.

The module is an extension of the original seed; it introduces a
- 1‑hidden‑layer MLP that learns a feature map from the raw input,
- 2‑layer variational quantum circuit with 2‑qubit entangling gates,
- a shared linear head that consumes both classical and quantum features,
- dropout for uncertainty quantification.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Data generation – identical to the original seed but with a slightly
# different random‑angle distribution for reproducibility.
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data that is a superposition of two Gaussian
    components with random angles.  The seed is the same as the original
    reference but the random‑angle distribution is changed to a
    uniform distribution over [0, 2π)."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = np.random.uniform(0, 2 * np.pi, size=samples)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that mirrors the original seed but uses the new data
    generator.  The data are returned as tensors suitable for
    feeding into a PyTorch model."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Hybrid model – classical MLP + quantum module
# --------------------------------------------------------------------------- #
class HybridQModel(nn.Module):
    """Hybrid classical‑quantum regression model.

    The model consists of:
        * A classical MLP that learns a linear embedding of the input.
        * A quantum module (imported from the QML module) that produces
          a feature vector via a variational circuit.
        * A shared linear head that consumes the concatenated features
          and produces a scalar output.
    Dropout is applied to the classical features to allow
    Monte‑Carlo dropout for uncertainty estimation.
    """
    def __init__(self, num_features: int, num_wires: int, dropout_prob: float = 0.1):
        super().__init__()
        # Classical branch
        self.classical = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        # Quantum branch – import the quantum module from the QML file
        from.QuantumRegression__gen271_qml import HybridQuantumModule  # noqa: E402
        self.quantum = HybridQuantumModule(num_wires)
        # Shared head
        self.head = nn.Linear(16 + num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Classical features
        class_feat = self.classical(state_batch)
        # Quantum features
        qdev = self.quantum(state_batch)
        # Concatenate and produce output
        combined = torch.cat([class_feat, qdev], dim=-1)
        return self.head(combined).squeeze(-1)

__all__ = ["HybridQModel", "RegressionDataset", "generate_superposition_data"]
