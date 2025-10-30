"""Hybrid classical-quantum estimator combining feed‑forward and variational layers."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample qubit‑pair superpositions for regression."""
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
    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset returning quantum states and regression targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridEstimatorQNN(nn.Module):
    """
    Classical encoder that translates input features into parameters for a quantum circuit
    and a classical head that consumes expectation values.
    """
    def __init__(self, num_features: int, num_wires: int, num_weight_params: int):
        super().__init__()
        self.num_wires = num_wires
        # Encoder maps features to 2*num_wires rotation angles
        self.input_encoder = nn.Sequential(
            nn.Linear(num_features, 4 * num_wires),
            nn.ReLU(),
            nn.Linear(4 * num_wires, 2 * num_wires),
        )
        # Encoder maps features to variational weight parameters
        self.weight_encoder = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, num_weight_params),
        )
        # Classical head maps quantum expectation values to a scalar output
        self.head = nn.Linear(num_wires, 1)

    def forward(self, features: torch.Tensor, quantum_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor
            Classical input features of shape (batch, num_features).
        quantum_features : torch.Tensor
            Expectation values from the quantum circuit of shape (batch, num_wires).

        Returns
        -------
        torch.Tensor
            Regression predictions.
        """
        return self.head(quantum_features).squeeze(-1)

    def get_params(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Produce rotation angles and weight parameters for a quantum circuit from
        classical features.

        Returns
        -------
        input_params, weight_params : tuple[torch.Tensor, torch.Tensor]
            Each of shape (batch, *param_shape).
        """
        input_params = self.input_encoder(features)          # (batch, 2 * num_wires)
        weight_params = self.weight_encoder(features)        # (batch, num_weight_params)
        return input_params, weight_params

__all__ = ["HybridEstimatorQNN", "RegressionDataset", "generate_superposition_data"]
