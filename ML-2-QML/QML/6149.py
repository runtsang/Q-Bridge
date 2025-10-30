"""Unified quantum regression model with an encoder, variational layer, and linear read‑out.

The model mirrors the classical architecture: a Ry‑rotation encoder that maps the input state
to a superposition, a variational layer with random gates and single‑qubit rotations,
and a linear head that projects the Pauli‑Z expectations to a scalar output.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

__all__ = ["RegressionDataset", "generate_superposition_data", "QuantumRegression"]

# --------------------------------------------------------------------------- #
# Dataset and data‑generation utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data that mimics the quantum superposition model."""
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
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that yields quantum states and target values."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Quantum encoder and variational layer
# --------------------------------------------------------------------------- #
class QEncoder(tq.QuantumModule):
    """Ry‑rotation encoder that maps input features to a superposition."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice, state_batch: torch.Tensor):
        qdev.set_state(state_batch)
        for wire in range(self.n_wires):
            self.ry(qdev, wires=wire)

class QVariationalLayer(tq.QuantumModule):
    """Random layer + trainable single‑qubit rotations."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

# --------------------------------------------------------------------------- #
# Quantum regression head
# --------------------------------------------------------------------------- #
class QuantumRegression(tq.QuantumModule):
    """Full quantum regression model."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = QEncoder(num_wires)
        self.var_layer = QVariationalLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=state_batch.device,
        )
        self.encoder(qdev, state_batch)
        self.var_layer(qdev)
        expectations = self.measure(qdev)
        return self.head(expectations).squeeze(-1)
