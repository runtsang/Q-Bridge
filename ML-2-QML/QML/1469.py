"""Quantum regression model with a deep variational circuit and a classical read‑out head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

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

class HybridRegressionDataset(torch.utils.data.Dataset):
    """Same synthetic dataset as the classical counterpart but returns complex states."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridRegressor(tq.QuantumModule):
    """A VQC with multiple entangling layers and a classical read‑out head."""
    class EntangleLayer(tq.QuantumModule):
        def __init__(self, wires: list[int]):
            super().__init__()
            self.wires = wires
            self.cnot = tq.CNOT

        def forward(self, qdev: tq.QuantumDevice):
            for i in range(len(self.wires) - 1):
                self.cnot(qdev, wires=[self.wires[i], self.wires[i + 1]])

    class ParameterizedLayer(tq.QuantumModule):
        def __init__(self, wires: list[int]):
            super().__init__()
            self.wires = wires
            self.rx = tq.RX(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            for w in self.wires:
                self.rx(qdev, wires=w)
                self.rz(qdev, wires=w)

    def __init__(self, num_wires: int, num_layers: int = 3):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(self.ParameterizedLayer(list(range(num_wires))))
            self.layers.append(self.EntangleLayer(list(range(num_wires))))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        for layer in self.layers:
            layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressor", "HybridRegressionDataset"]
