"""Quantum regression model using Pennylane with amplitude embedding and variational ansatz."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data in the amplitude basis."""
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    # Construct normalized amplitude vectors of length 2**n
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        amp0 = np.cos(thetas[i])
        amp1 = np.exp(1j * phis[i]) * np.sin(thetas[i])
        states[i, 0] = amp0
        states[i, -1] = amp1
    # Labels derived from the underlying parameters with noise
    labels = np.sin(2 * thetas) * np.cos(phis) + 0.05 * np.random.randn(samples)
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding amplitude-encoded state vectors."""

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
    """Hybrid quantum-classical regression model with Pennylane."""

    def __init__(self, num_wires: int, dev: qml.Device | None = None):
        super().__init__()
        self.num_wires = num_wires
        self.dev = dev or qml.device("default.qubit", wires=num_wires, shots=1024)

        def circuit(params, state):
            # Amplitude embedding
            qml.AmplitudeEmbedding(features=state, wires=range(num_wires), normalize=True)
            # Parameterised rotation layer
            for i in range(num_wires):
                qml.RY(params[i], wires=i)
            # Entangling layer
            for i in range(num_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation values of Z on each wire
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_wires)]

        self.qnode = qml.QNode(circuit, self.dev, interface="torch")

        # Classical head to map expectation values to a scalar
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        batch_size = state_batch.shape[0]
        # Allocate parameter tensors (one per qubit)
        params = torch.randn(batch_size, self.num_wires, requires_grad=True, device=state_batch.device)
        # Run the quantum circuit for each sample in the batch
        out = torch.stack(
            [self.qnode(params[i], state_batch[i]) for i in range(batch_size)], dim=0
        )
        # Expectation values of Z on each wire
        features = out
        return self.head(features).squeeze(-1)


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
