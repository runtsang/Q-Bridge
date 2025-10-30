"""QuantumRegression__gen282 module – quantum regression with Pennylane.

The QML model now uses:
* a custom feature map based on Ry rotations,
* a trainable variational layer,
* batch‑aware forward pass,
* a linear head that maps expectation values to a scalar output.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pennylane as qml
from pennylane import numpy as pnp

def generate_superposition_data(
    num_wires: int,
    samples: int,
    *,
    noise: float = 0.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a superposition‑based quantum dataset.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the state.
    samples : int
        Number of samples to generate.
    noise : float, optional
        Standard deviation of additive Gaussian noise added to targets.
    random_state : int | None, optional
        Seed for reproducibility.
    """
    rng = np.random.default_rng(random_state)
    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    if noise > 0.0:
        labels += rng.normal(scale=noise, size=labels.shape).astype(np.float32)
    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Torch dataset that yields quantum states and scalar targets."""
    def __init__(self, samples: int, num_wires: int, noise: float = 0.0, random_state: int | None = None):
        self.states, self.labels = generate_superposition_data(
            num_wires, samples, noise=noise, random_state=random_state
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """Hybrid quantum‑classical regression model using Pennylane."""
    def __init__(self, num_wires: int, num_layers: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_wires)
        # Learnable parameters for each layer and wire
        self.params = nn.Parameter(torch.randn(num_layers, num_wires))
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")
        self.head = nn.Linear(num_wires, 1)

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Variational circuit that encodes `x` and applies trainable rotations."""
        # Feature map: Ry rotations
        for i, wire in enumerate(range(self.num_wires)):
            qml.RY(x[i], wires=wire)
        # Variational layers
        for layer in range(self.num_layers):
            for wire in range(self.num_wires):
                qml.RX(params[layer, wire], wires=wire)
                qml.RY(params[layer, wire], wires=wire)
            for wire in range(self.num_wires - 1):
                qml.CNOT(wires=[wire, wire + 1])
        # Return expectation values of PauliZ on each wire
        return [qml.expval(qml.PauliZ(w)) for w in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Compute predictions for a batch of quantum states."""
        batch_size = state_batch.shape[0]
        # For simplicity, use the first `num_wires` amplitudes as features
        features = state_batch[:, :self.num_wires]
        # Compute expectation values for each sample
        expvals = []
        for i in range(batch_size):
            expvals.append(self.qnode(features[i], self.params))
        expvals = torch.stack(expvals, dim=0)
        return self.head(expvals).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
