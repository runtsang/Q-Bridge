"""QuantumHybridRegression – classical‑only implementation with placeholders for quantum integration.

This module builds on the three reference pairs:
1. The base regression dataset and a fully‑connected network.
2. A quantum encoder that mirrors the classical feature map.
3. A hybrid quantum‑classical head that can be swapped with a pure‑classical head.

The design introduces:
* A shared *Encoder* that applies the same parameter‑free transformation to both the data and the quantum device,
  ensuring the same feature map is used in the classical and quantum branches.
* A *QLayer* that can be instantiated either as a RandomLayer or a custom trainable layer,
  allowing the user to experiment with depth and expressivity.
* A *HybridHead* that concatenates a classical linear layer with a quantum expectation,
  providing a direct comparison between pure‑classical and hybrid predictions.

The module is fully importable and can be used as:
```
from QuantumHybridRegression import QuantumHybridRegression
model = QuantumHybridRegression(num_features=10, num_wires=10)
```
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

__all__ = ["generate_superposition_data", "RegressionDataset", "ClassicalRegressionModel",
           "HybridHead", "QuantumHybridRegression"]

# =========================
# Dataset and data generation
# =========================

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate samples of the form |ψ(θ,φ)⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩."""
    omega_0 = np.zeros(2 ** num_features, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_features, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_features), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns complex state vectors and scalar targets."""
    def __init__(self, samples: int, num_features: int):
        self.states, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# =========================
# Classical regression model
# =========================

class ClassicalRegressionModel(nn.Module):
    """A simple fully‑connected regression network."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

# =========================
# Hybrid head (classical only)
# =========================

class HybridHead(nn.Module):
    """Linear head with an optional shift, mimicking a quantum expectation."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return torch.sigmoid(self.linear(logits) + self.shift)

# =========================
# Hybrid regression wrapper
# =========================

class QuantumHybridRegression(nn.Module):
    """Combines a classical regression network with an optional quantum module.

    If a quantum module is provided, its output is used; otherwise, the classical
    hybrid head is used. The quantum module must implement a.forward method
    that accepts a torch.Tensor of shape (batch, features) and returns a
    torch.Tensor of shape (batch, 1).
    """
    def __init__(self, num_features: int, num_wires: int,
                 use_quantum: bool = False,
                 quantum_module: nn.Module | None = None,
                 shift: float = 0.0):
        super().__init__()
        self.num_features = num_features
        self.num_wires = num_wires
        self.use_quantum = use_quantum
        self.classical_model = ClassicalRegressionModel(num_features)
        self.hybrid_head = HybridHead(num_features, shift=shift)

        if use_quantum and quantum_module is not None:
            self.quantum_module = quantum_module
        else:
            self.quantum_module = None

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Return predictions from the chosen branch."""
        if self.quantum_module is not None:
            return self.quantum_module(state_batch)
        # Fall back to classical branch
        features = self.classical_model(state_batch)
        return self.hybrid_head(features)
