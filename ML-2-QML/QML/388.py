"""Quantum regression model with a variational circuit and classical post-processing.

This module extends the original seed by adding a parameterized entangling layer,
a custom encoder that uses a feature map based on the number of wires, and a
classical post-processing head that outputs both target and confidence.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states and labels."""
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
    """Dataset that returns quantum states and labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """Quantum regression model with a variational circuit and confidence head."""
    class EntanglingLayer(tq.QuantumModule):
        """A simple entangling layer using CNOTs."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.cnot = tq.CNOT(wires=[(i, (i + 1) % n_wires) for i in range(n_wires)])

        def forward(self, qdev: tq.QuantumDevice):
            self.cnot(qdev)

    class VariationalBlock(tq.QuantumModule):
        """Parameterised rotation block."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder using a feature map that scales with the number of wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.entangle = self.EntanglingLayer(num_wires)
        self.varblock = self.VariationalBlock(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head_target = nn.Linear(num_wires, 1)
        self.head_confidence = nn.Sequential(
            nn.Linear(num_wires, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.entangle(qdev)
        self.varblock(qdev)
        features = self.measure(qdev)
        target = self.head_target(features).squeeze(-1)
        confidence = self.head_confidence(features).squeeze(-1)
        return target, confidence

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
