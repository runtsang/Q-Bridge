"""Hybrid quantum‑classical regression model using Pennylane.

Key extensions over the seed:
- Flexible data encoding (amplitude or angle).
- Variational circuit with configurable depth and entanglement.
- Classical head with a linear layer for regression.
- Parameters are registered as PyTorch tensors for seamless training.
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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

class RegressionDataset(Dataset):
    """Dataset that returns complex quantum states and regression labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """Hybrid quantum‑classical regression model."""
    def __init__(
        self,
        num_wires: int,
        encoding: str = "amplitude",
        depth: int = 3,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_wires = num_wires
        self.encoding = encoding
        self.depth = depth

        # Quantum device
        self.qdev = qml.device("default.qubit", wires=num_wires, shots=None)

        # Variational parameters
        self.weight_params = nn.Parameter(torch.randn(depth, num_wires, requires_grad=True))

        # Quantum node
        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Encoding
            if self.encoding == "amplitude":
                qml.QubitStateVector(inputs, wires=range(num_wires))
            elif self.encoding == "angle":
                for i in range(num_wires):
                    qml.RX(inputs[i], wires=i)
            else:
                raise ValueError(f"Unknown encoding {self.encoding}")

            # Variational layers
            for d in range(depth):
                for i in range(num_wires):
                    qml.RY(weights[d, i], wires=i)
                for i in range(num_wires):
                    qml.CNOT(wires=[i, (i + 1) % num_wires])

            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.circuit = circuit

        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # state_batch: (batch, 2**n) complex
        batch_size = state_batch.shape[0]
        outputs = []
        for i in range(batch_size):
            out = self.circuit(state_batch[i], self.weight_params)
            outputs.append(out)
        features = torch.stack(outputs, dim=0)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
