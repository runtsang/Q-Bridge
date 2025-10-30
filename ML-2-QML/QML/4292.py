"""Hybrid quantum‑classical regression model.

The quantum branch implements a convolutional encoder that maps a 2‑D image
into a set of qubits, followed by a parameterised quantum layer, a measurement
stage, and a classical linear head.  The design follows the *combination*
scaling paradigm: the quantum circuit is wrapped inside a classical
convolutional pre‑processor, allowing the model to benefit from both
quantum feature extraction and classical regression.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data generation and dataset
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data in the form of complex amplitudes."""
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

class RegressionDataset(Dataset):
    """Dataset that returns a complex state vector and a target."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Quantum convolutional encoder
# --------------------------------------------------------------------------- #
class QuantumConvEncoder(tq.QuantumModule):
    """Encode a 2‑D image into a register of qubits."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )

    def forward(self, qdev: tq.QuantumDevice, states: torch.Tensor) -> None:
        # states shape: (batch, 1, side, side)
        flat = states.view(states.shape[0], -1)
        self.encoder(qdev, flat)

# --------------------------------------------------------------------------- #
# Quantum fully‑connected layer
# --------------------------------------------------------------------------- #
class QuantumFullyConnectedLayer(tq.QuantumModule):
    """Parameterised quantum layer with random gates and single‑qubit rotations."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

# --------------------------------------------------------------------------- #
# Hybrid quantum‑classical regression model
# --------------------------------------------------------------------------- #
class QModel(tq.QuantumModule):
    """Hybrid quantum‑classical regression model."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = QuantumConvEncoder(num_wires)
        self.q_layer = QuantumFullyConnectedLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # state_batch shape: (batch, 1, side, side)
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
