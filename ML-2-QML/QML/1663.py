"""Quantum regression model built with PennyLane."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as npnp


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create superposition states |ψ⟩ = cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩."""
    thetas = 2 * npnp.pi * npnp.random.rand(samples)
    phis = 2 * npnp.pi * npnp.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = npnp.cos(thetas[i]) * npnp.array([1] + [0] * (2 ** num_wires - 1), dtype=complex) + \
                    npnp.exp(1j * phis[i]) * npnp.sin(thetas[i]) * npnp.array([0] * (2 ** num_wires - 1) + [1], dtype=complex)
    labels = npnp.sin(2 * thetas) * npnp.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding quantum states and regression targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class SharedClassName(nn.Module):
    """Hybrid variational regression model using PennyLane."""

    def __init__(self, num_wires: int, n_layers: int = 4, n_params: int = 3):
        super().__init__()
        self.num_wires = num_wires
        dev = qml.device("default.qubit", wires=num_wires)

        @qml.qnode(dev, interface="torch")
        def circuit(state, params):
            # Encode classical data into rotation angles
            for i, wire in enumerate(range(num_wires)):
                qml.RY(state[i], wires=wire)
            # Entangling block
            for _ in range(n_layers):
                for wire in range(num_wires - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                for wire in range(num_wires):
                    qml.RY(params[wire], wires=wire)
            # Measure all Pauli‑Z
            return qml.expval(qml.PauliZ(wires=range(num_wires)))

        self.circuit = circuit
        # Linear head maps expectation values to scalar output
        self.head = nn.Linear(num_wires, 1)
        # Trainable parameters for the variational part
        self.params = nn.Parameter(torch.randn(num_wires))

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # state_batch shape: (batch, wires)
        expectations = torch.stack([self.circuit(state, self.params) for state in state_batch])
        return self.head(expectations).squeeze(-1)


__all__ = ["SharedClassName", "RegressionDataset", "generate_superposition_data"]
