"""Hybrid regression model – quantum implementation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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
    """Dataset for quantum regression."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QSampler(tq.QuantumModule):
    """Quantum sampler layer that mirrors the classical SamplerQNN."""

    def __init__(self, n_wires: int = 2):
        super().__init__()
        self.n_wires = n_wires
        # Trainable rotation parameters for each qubit
        self.weight_params = nn.Parameter(torch.randn(n_wires * 2) * 0.01)
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        # Apply parameterized rotations
        for i in range(self.n_wires):
            self.rx(qdev, wires=i, params=self.weight_params[i])
            self.ry(qdev, wires=i, params=self.weight_params[self.n_wires + i])
        # Measure all qubits in the Z basis
        return tq.MeasureAll(tq.PauliZ)(qdev)  # shape: (bsz, n_wires)


class QLayer(tq.QuantumModule):
    """Intermediate quantum layer with random operations and rotations."""

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


class HybridRegressionModel(tq.QuantumModule):
    """Hybrid quantum regression model that combines an encoder, QLayer, QSampler, and a classical head."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder maps input state vectors to a higher‑dimensional representation
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = QLayer(num_wires)
        self.sampler = QSampler(n_wires=2)  # 2‑qubit sampler for feature extraction
        self.head = nn.Linear(2, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        sampler_features = self.sampler(qdev)  # shape: (bsz, 2)
        return self.head(sampler_features).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
