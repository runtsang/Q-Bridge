"""Quantum regression model using Pennylane variational circuits and a classical head."""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data using a superposition-inspired function with added Gaussian noise."""
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
    noise = np.random.normal(scale=0.05, size=labels.shape).astype(np.float32)
    return states, (labels + noise).astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class RegressionModel(nn.Module):
    def __init__(self, num_wires: int, num_layers: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.device = qml.device("default.qubit", wires=num_wires)
        # Trainable parameters for the variational circuit
        self.params = nn.Parameter(torch.randn(num_layers, num_wires, 3))
        # Classical head
        self.head = nn.Linear(num_wires, 1)

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(state, params):
            qml.StatePrep(state, wires=range(num_wires))
            qml.templates.StronglyEntanglingLayers(params, wires=range(num_wires))
            return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

        self.circuit = circuit

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # state_batch: (batch_size, 2**num_wires)
        expectations = torch.stack([self.circuit(state, self.params) for state in state_batch])
        return self.head(expectations).squeeze(-1)


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
