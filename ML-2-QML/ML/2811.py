"""Hybrid classical‑quantum regression model and dataset.

The module contains:
* `generate_superposition_data` – creates complex superposition states and smooth labels.
* `RegressionDataset` – PyTorch Dataset yielding state tensors and targets.
* `HybridModel` – a pure‑classical MLP that accepts both classical features (derived from state amplitudes)
  and pre‑computed quantum features (from the QML module).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data: a superposition of |0…0⟩ and |1…1⟩ with random angles.

    The returned *features* are the complex amplitudes of the quantum states,
    while *labels* are a smooth function of the underlying angles.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    # Build the two basis states
    omega_0 = np.zeros(2 ** num_features, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_features, dtype=complex)
    omega_1[-1] = 1.0

    states = np.cos(thetas[:, None]) * omega_0 + np.exp(1j * phis[:, None]) * np.sin(thetas[:, None]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)

    return states.astype(np.complex64), labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that yields a complex state tensor and the regression target."""

    def __init__(self, samples: int, num_features: int):
        self.states, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridModel(nn.Module):
    """Classical‑quantum hybrid regression model.

    The model first extracts a classical feature from the state amplitudes,
    then concatenates it with a quantum feature vector produced by a separate QML
    module.  The concatenated vector is passed through a final linear head.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

        # Classical stream – simple MLP on amplitude magnitude
        self.classical_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Final regression head
        self.final_head = nn.Linear(16 + 1, 1)

    def forward(self, state_batch: torch.Tensor, quantum_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Complex tensor of shape (batch, 2**num_features).
        quantum_features : torch.Tensor
            Quantum feature vector of shape (batch, 1) produced by the QML module.
        """
        # Classical feature: magnitude of the first amplitude (or any other single value)
        classical_feat = torch.abs(state_batch[:, 0:1]).to(torch.float32)
        x_c = self.classical_net(classical_feat)

        # Concatenate classical and quantum streams
        x = torch.cat([x_c, quantum_features], dim=1)
        return self.final_head(x).squeeze(-1)


__all__ = ["HybridModel", "RegressionDataset", "generate_superposition_data"]
