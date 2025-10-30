"""Quantum regression model using a hardware‑efficient ansatz and amplitude encoding.

Features:
* Amplitude‑encoding of complex inputs.
* Parameterised rotational layers followed by entangling gates.
* A classical linear head for regression.
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and sinusoidal targets."""
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
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset providing amplitude‑encoded states and labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class RegressionModel(nn.Module):
    """Hybrid quantum‑classical regression model."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        # Classical head
        self.head = nn.Linear(num_wires, 1)
        # Quantum device
        self.dev = qml.device("default.qubit", wires=num_wires, shots=1)
        # Trainable parameters for the variational ansatz
        self.param_shape = (2 * num_wires,)
        self.params = nn.Parameter(torch.randn(self.param_shape))
        # Quantum node
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(x: torch.Tensor, params: torch.Tensor):
            # Amplitude encoding
            qml.QubitStateVector(x, wires=range(num_wires))
            # Variational layers
            for i in range(num_wires):
                qml.RX(params[2 * i], wires=i)
                qml.RY(params[2 * i + 1], wires=i)
            # Entangling layer
            for i in range(num_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]
        self.qnode = qnode

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Execute the quantum circuit for each sample and apply the classical head."""
        # Compute quantum features for each batch element
        batch_features = torch.stack([self.qnode(state, self.params) for state in state_batch])
        return self.head(batch_features).squeeze(-1)


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
