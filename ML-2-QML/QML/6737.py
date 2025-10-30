"""Quantum regression dataset and model with entanglement and multi‑head readout.

The quantum circuit is built with Pennylane and includes a trainable rotation layer,
a ladder of CNOTs and parameterised Z‑rotations.  The measurement yields a vector
of Pauli‑Z expectation values that is fed into a linear readout head.
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic superposition states for regression.
    """
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
    """Dataset wrapping the synthetic quantum states."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModelGen595(nn.Module):
    """Variational quantum circuit for regression."""
    def __init__(self, num_wires: int, device: str = "default.qubit"):
        super().__init__()
        self.num_wires = num_wires
        self.dev = qml.device(device, wires=num_wires)
        # trainable parameters
        self.theta = nn.Parameter(torch.randn(num_wires))
        self.phi = nn.Parameter(torch.randn(num_wires - 1))
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        batch_size = state_batch.shape[0]
        outputs = []
        for i in range(batch_size):
            state = state_batch[i].cpu().numpy()
            @qml.qnode(self.dev, interface="torch")
            def circuit():
                qml.QubitStateVector(state, wires=range(self.num_wires))
                for w in range(self.num_wires):
                    qml.RY(self.theta[w], wires=w)
                for w in range(self.num_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
                    qml.RZ(self.phi[w], wires=w + 1)
                return qml.expval(qml.PauliZ(wires=range(self.num_wires)))
            outputs.append(circuit())
        out = torch.stack(outputs)
        return self.head(out).squeeze(-1)

__all__ = ["QModelGen595", "RegressionDataset", "generate_superposition_data"]
