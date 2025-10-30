"""
Quantum regression model with a parameter‑efficient ansatz and a quantum kernel estimator.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    The labels are a non‑linear function of theta and phi.
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
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that wraps the quantum synthetic data into a PyTorch Dataset.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QLayer(tq.QuantumModule):
    """
    A lightweight variational layer that applies a random layer followed by
    trainable single‑qubit rotations. The number of parameters is kept
    small to encourage efficient training.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Random layer with only 10 operations for parameter efficiency
        self.random_layer = tq.RandomLayer(n_ops=10, wires=list(range(num_wires)))
        # Trainable single‑qubit rotations
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

class QModel(tq.QuantumModule):
    """
    Quantum regression model that encodes the input state, applies a
    parameter‑efficient variational layer, measures all qubits in the
    Pauli‑Z basis and projects the resulting expectation values to a
    scalar output via a classical linear head.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps the input state to the quantum device
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Parameter‑efficient variational layer
        self.q_layer = QLayer(num_wires)
        # Measurement of all qubits in the Pauli‑Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head that maps expectation values to a scalar
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that creates a quantum device, encodes the batch,
        applies the variational layer, measures, and then projects
        to a scalar output.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        out = self.head(features)
        return out.squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
