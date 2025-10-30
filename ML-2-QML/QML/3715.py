"""Quantum regression model inspired by QCNN architecture using torchquantum."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate dataset of superposition states cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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
    """Dataset for quantum regression tasks."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ConvUnit(tq.QuantumModule):
    """Two‑qubit convolution unit used in QCNN."""
    def __init__(self):
        super().__init__()
        self.rz1 = tq.RZ(has_params=True, trainable=True)
        self.cx = tq.CX()
        self.rz2 = tq.RZ(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice, wires: tuple[int, int]) -> None:
        self.rz1(qdev, wires=wires[0])
        self.cx(qdev, wires=wires)
        self.rz2(qdev, wires=wires[0])
        self.ry(qdev, wires=wires[1])
        self.cx(qdev, wires=wires)
        self.rz1(qdev, wires=wires[0])  # reuse for symmetry


class PoolUnit(tq.QuantumModule):
    """Two‑qubit pooling unit used in QCNN."""
    def __init__(self):
        super().__init__()
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.cx = tq.CX()
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice, wires: tuple[int, int]) -> None:
        self.rz(qdev, wires=wires[0])
        self.cx(qdev, wires=wires)
        self.ry(qdev, wires=wires[1])
        self.cx(qdev, wires=wires)
        self.rz(qdev, wires=wires[0])  # for symmetry


class QCNNQuantum(tq.QuantumModule):
    """Quantum circuit implementing the QCNN architecture."""
    def __init__(self, num_qubits: int):
        super().__init__()
        self.n_qubits = num_qubits
        self.conv1 = ConvUnit()
        self.pool1 = PoolUnit()
        self.conv2 = ConvUnit()
        self.pool2 = PoolUnit()
        self.conv3 = ConvUnit()
        self.pool3 = PoolUnit()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_qubits, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=state_batch.device)

        # Feature map: simple Ry encoding
        tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{self.n_qubits}xRy"])(qdev, state_batch)

        # First convolution block
        for i in range(0, self.n_qubits, 2):
            self.conv1(qdev, (i, i + 1))
        self.pool1(qdev, (0, 1))

        # Second convolution block
        for i in range(2, self.n_qubits, 2):
            self.conv2(qdev, (i, i + 1))
        self.pool2(qdev, (2, 3))

        # Third convolution block
        for i in range(4, self.n_qubits, 2):
            self.conv3(qdev, (i, i + 1))
        self.pool3(qdev, (4, 5))

        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["generate_superposition_data", "RegressionDataset", "QCNNQuantum"]
