"""QuantumRegression__gen236_qml.py

Quantum regression module based on PennyLane, featuring a variational circuit with
rotation and CNOT entanglement and a linear read‑out head.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pennylane as qml

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum states of the form cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩.

    The labels are a nonlinear function of the angles, mirroring the original
    seed but with a more compact implementation.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    states = np.cos(thetas)[:, None] * omega_0 + np.exp(1j * phis)[:, None] * np.sin(thetas)[:, None] * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset class for quantum regression."""

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
    """Hybrid quantum‑classical regression model based on PennyLane.

    The circuit consists of a classical encoding via ``RY`` gates, followed by
    ``num_layers`` variational blocks of ``Rot`` rotations and nearest‑neighbour CNOT
    entanglement.  The expectation values of ``PauliZ`` on each wire are fed into
    a linear head.
    """

    def __init__(self, num_wires: int, num_layers: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers

        # Trainable variational parameters
        self.params = nn.Parameter(torch.randn(num_layers, num_wires, 3))

        # Linear read‑out
        self.head = nn.Linear(num_wires, 1)

        # PennyLane device and QNode
        self.dev = qml.device("default.qubit", wires=num_wires)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x, params):
        """Quantum circuit for a single sample."""
        # Classical encoding
        for i, wire in enumerate(range(self.num_wires)):
            qml.RY(x[i], wires=wire)

        # Variational layers
        for layer in range(self.num_layers):
            for i, wire in enumerate(range(self.num_wires)):
                qml.Rot(*params[layer, i], wires=wire)
            # Entanglement
            for i in range(self.num_wires - 1):
                qml.CNOT(wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Batch‑wise forward pass."""
        outputs = torch.stack([self.qnode(x, self.params) for x in state_batch], dim=0)
        return self.head(outputs).squeeze(-1)
