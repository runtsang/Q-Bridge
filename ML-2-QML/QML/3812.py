"""Hybrid quantum regression model combining encoding, sampler circuit, and a classical head."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(θ)|0…0⟩ + e^{iϕ} sin(θ)|1…1⟩.
    The label is y = sin(2θ) * cos(ϕ).
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

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding (state, target) pairs for quantum regression."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QSampler(tq.QuantumModule):
    """
    Parameterized quantum sampler circuit.
    Consists of a random layer followed by trainable RX/RY rotations on each wire.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.num_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

class HybridRegression(tq.QuantumModule):
    """
    Quantum hybrid regression model:
    * GeneralEncoder encodes classical data into a superposition.
    * QSampler applies a trainable variational circuit.
    * Measurement of Pauli‑Z on all wires yields expectation values.
    * A classical linear head maps the expectation vector to a scalar output.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.qsampler = QSampler(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.qsampler(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data", "QSampler"]
